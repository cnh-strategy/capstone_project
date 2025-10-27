import json
import os
from collections import defaultdict
from typing import Optional, List, Dict

import requests
import torch
import yfinance as yf
from typing import Dict, List, Optional, Literal, Tuple, Any
import warnings
from dataclasses import dataclass, field
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from agents.dump import CAPSTONE_OPENAI_API
from agents.macro_shap_llm import LLMExplainer, AttributionAnalyzer
from agents.macro_sub import get_std_pred
from debate_ver4.prompts import REBUTTAL_PROMPTS
from debate_ver4.config.agents import dir_info, agents_info

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 공통 설정
# ============================================================
OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = "2020-01-01"
END_DATE = '2024-12-31'

save_dir=dir_info["data_dir"]
model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]



@dataclass
class Target:
    """예측 목표값 + 불확실성 정보 포함
    - next_close: 다음 거래일 종가 예측치
    - uncertainty: Monte Carlo Dropout 기반 예측 표준편차(σ)
    - confidence: 모델 신뢰도 β (정규화된 신뢰도; 선택적)
    """
    next_close: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None

@dataclass
class Opinion:
    agent_id: str
    target: Target
    reason: str

@dataclass
class Rebuttal:
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str

@dataclass
class RoundLog:
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    """에이전트 입력 원천 데이터(필요 시 자유 확장)
    - sentimental: 심리/커뮤니티/뉴스 스냅샷
    - fundamental: 재무/밸류에이션 요약
    - technical  : 가격/지표 스냅샷
    - last_price : 최신 종가
    - currency   : 통화코드
    """
    SentimentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    FundamentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    TechnicalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    last_price: Optional[float] = None
    currency: Optional[str] = None


# ============================================================
# MacroSentimentAgent — 시장·거시경제 시계열 기반
# ============================================================
class MacroSentimentAgentDataset:
    OPENAI_URL = "https://api.openai.com/v1/responses"

    def __init__(self,
                 agent_id="MacroSentiAgent",
                 preferred_models: Optional[List[str]] = None,
                 option_model: Optional[str] = None,
                 verbose: bool = False,
                 temperature: float = 0.2,
                 ticker = None,
                 **kwargs
                 ):
        self.temperature = None
        self.target = None
        self.opinions = None
        self.macro_tickers = {
            "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
            "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX",
            "^VIX": "^VIX",
            "DX-Y.NYB": "DX-Y.NYB",
            "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
            "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F",
          #  "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"
        }
        self.data = None
        self.agent_id = 'MacroSentiAgent'
        self.ticker_name = ticker


        self.tickers = [self.ticker_name] or ["AAPL", "MSFT", "NVDA"]
        # self.target_tickers = target_tickers or ["AAPL", "MSFT", "NVDA"]

        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.macro_df = None
        self.pred_df = None
        self.X_scaled = None

        self.window_size = 40
        # 모델 폴백 우선순위
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if option_model:
            self.preferred_models = [option_model] + [
                m for m in self.preferred_models if m != option_model
            ]

        # 공통 헤더
        self.api_key = CAPSTONE_OPENAI_API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.temperature = temperature # Temperature 설정
        self.verbose = verbose            # 디버깅 모드
        self.rebuttals: Dict[int, List[Rebuttal]] = defaultdict(list)

    def fetch_data(self):
        """다중 티커 데이터 다운로드"""
        df = yf.download(
            tickers=list(self.macro_tickers.values()),
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            group_by="ticker",
            auto_adjust=False
        )

        # ✅ pandas 버전/구조 관계없이 일관된 포맷으로 변환
        # MultiIndex 구조일 경우 (티커별로 OHLCV 존재)
        if isinstance(df.columns, pd.MultiIndex):
            # 구조를 (날짜, 티커, 값) 형태로 변환
            df = df.stack(level=0)
            df.index.names = ["Date", "Ticker"]
            df.sort_index(inplace=True)

            # 컬럼 이름 평탄화
            df.columns = [col for col in df.columns]
            df = df.unstack(level="Ticker")
            df.columns = ["_".join(col).strip() for col in df.columns.values]
        else:
            # 단일 인덱스 구조인 경우 그대로 사용
            df.index.name = "Date"

        self.data = df
        print(f"[MacroSentimentAgent] Data shape: {df.shape}, Columns: {len(df.columns)}")
        return df

    def add_features(self):
        """수익률, 금리차, 위험심리 등 계산"""
        df = self.data.copy()

        # 각 자산의 1일 수익률
        for ticker in self.macro_tickers.values():
            if (ticker, "Close") in df.columns:
                df[(ticker, "ret_1d")] = df[(ticker, "Close")].pct_change()

        # 금리 스프레드 (10년 - 3개월)
        if ("^TNX", "Close") in df.columns and ("^IRX", "Close") in df.columns:
            df[("macro", "Yield_spread")] = df[("^TNX", "Close")] - df[("^IRX", "Close")]

        # 시장 위험심리 (SPY - DXY - VIX)
        if ("SPY", "ret_1d") in df.columns and ("DX-Y.NYB", "ret_1d") in df.columns and ("^VIX", "ret_1d") in df.columns:
            df[("macro", "Risk_Sentiment")] = (
                    df[("SPY", "ret_1d")] - df[("DX-Y.NYB", "ret_1d")] - df[("^VIX", "ret_1d")]
            )

        self.data = df
        return df

    def save_csv(self):
        path = os.path.join(OUTPUT_DIR, "macro_data/macro_sentiment.csv")
        self.data.to_csv(path, index=True)
        print(f"[MacroSentimentAgent] Saved {path}")


    def close_price_fetch(self, ticker_name):
        # 여러 종목의 일별 종가 불러오기 (2020-01-01 ~ 2024-12-31)
        df_prices = yf.download(
            ticker_name,
            start="2020-01-01",
            end="2025-01-03"
        )["Close"]

        # CSV 저장
        df_prices.to_csv(f"data/macro_data/daily_closePrice_{ticker_name}.csv")

        print("저장 완료:", df_prices.shape, "rows")

    # 주가와 매크로 데이터를 병합 + 최종 데이터셋 저장
    def make_dataset_seq(self, ticker_name):
        self.ticker_name = ticker_name
        # -------------------------------------------------------------
        # 1. 데이터 불러오기
        # -------------------------------------------------------------
        macro_df = pd.read_csv(f"data/macro_data/macro_sentiment.csv")
        price_df = pd.read_csv(f"data/macro_data/daily_closePrice_{ticker_name}.csv")

        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
        price_df['Date'] = pd.to_datetime(price_df['Date'])

        # -------------------------------------------------------------
        # 2. 매크로 피처 확장 (원본 + 변화율)
        # -------------------------------------------------------------
        macro_features = [c for c in macro_df.columns if c != 'Date']
        macro_ret = macro_df[macro_features].pct_change()
        macro_ret.columns = [f"{c}_ret" for c in macro_ret.columns]
        macro_full = pd.concat([macro_df, macro_ret], axis=1)
        macro_full = macro_full.replace([np.inf, -np.inf], np.nan).dropna(subset=['Date']).fillna(0)

        # -------------------------------------------------------------
        # 3. 주가 기반 피처 생성 (각 종목별)
        # -------------------------------------------------------------
        target_ticker_list = ['AAPL', 'MSFT', 'NVDA']   # ← 이름을 맞춤

        if ticker_name in price_df.columns:
            price_df[f"{ticker_name}_ret1"] = price_df[ticker_name].pct_change()
            price_df[f"{ticker_name}_ma5"] = price_df[ticker_name].rolling(5).mean()
            price_df[f"{ticker_name}_ma10"] = price_df[ticker_name].rolling(10).mean()
        else:
            print(f"[WARN] '{ticker_name}' column not found in price_df.columns: {price_df.columns.tolist()}")

        price_df = price_df.fillna(method='bfill')

        # -------------------------------------------------------------
        # 4. 날짜 기준 병합
        # -------------------------------------------------------------
        merged_df = pd.merge(price_df, macro_full, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
        print(f"[INFO] 병합 후 데이터 shape: {merged_df.shape}")

        # -------------------------------------------------------------
        # 5. Feature 선택
        # -------------------------------------------------------------
        macro_cols = [c for c in macro_full.columns if c != 'Date']
        price_cols = [c for c in merged_df.columns if any(t in c for t in target_ticker_list) and ('_ret' in c or '_ma' in c)]
        feature_cols = macro_cols + price_cols

        X_all = merged_df[feature_cols]

        # -------------------------------------------------------------
        # 6. 입력 스케일링
        # -------------------------------------------------------------
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_all)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        # -------------------------------------------------------------
        # 7. 타깃 (현재 ticker_name만 예측)
        # -------------------------------------------------------------
        if ticker_name in merged_df.columns:
            merged_df[f"{ticker_name}_target"] = merged_df[ticker_name].pct_change().shift(-1)
            y_all = merged_df[[f"{ticker_name}_target"]].dropna().reset_index(drop=True)
        else:
            print(f"[WARN] '{ticker_name}' not found in merged_df.columns: {merged_df.columns.tolist()}")
            return  # 혹은 raise Exception("Ticker not found in merged_df")

        X_scaled = X_scaled.iloc[:len(y_all)]

        # -------------------------------------------------------------
        # 8. 출력 스케일링
        # -------------------------------------------------------------
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = scaler_y.fit_transform(y_all)

        # -------------------------------------------------------------
        # 9. 시퀀스 생성 함수
        # -------------------------------------------------------------
        def create_sequences(X, y, window=40):
            Xs, ys = [], []
            for i in range(len(X) - window):
                Xs.append(X.iloc[i:(i + window)].values)
                ys.append(y[i + window])
            return np.array(Xs), np.array(ys)

        # -------------------------------------------------------------
        # 10. 시퀀스 변환
        # -------------------------------------------------------------
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=40)
        split_idx = int(len(X_seq) * 0.8)

        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]


        # -------------------------------------------------------------
        # 11. 최종 데이터 셋 저장
        # -------------------------------------------------------------
        csv_path = os.path.join(save_dir, f"{ticker_name}_{self.agent_id}_dataset.csv")
        merged_df.to_csv(csv_path, index=False)

        return X_train, X_test, y_train, y_test, X_seq, y_seq


    #StockData 및 X_tensor 생성 기능 (base_agent의 searcher 대응)
    def macro_searcher_add_funs(self, X_seq, feature_cols, agent_id=None):
        """
        searcher()의 마지막 단계와 동일한 기능:
        - 최신 윈도우 데이터(X_tensor) 생성
        - feature_dict 구성
        - StockData 초기화 및 값 저장
        """

        # ---------------------------------------------------------
        # 기본 정보 세팅
        # ---------------------------------------------------------
        if agent_id is None:
            agent_id = self.agent_id

        ticker_name = self.ticker_name or "UNKNOWN"

        # StockData 객체 생성
        self.stockdata = StockData()
        self.stockdata.ticker = ticker_name

        # ---------------------------------------------------------
        # 최신 윈도우(X_latest → X_tensor)
        # ---------------------------------------------------------
        X_latest = X_seq[-1:]              # 마지막 시퀀스 (예측용)
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        # DataFrame 변환
        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)

        # feature_dict 구성
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}

        # agent_id 이름으로 속성 추가 (예: self.stockdata.MacroSentiAgent)
        setattr(self.stockdata, agent_id, feature_dict)

        # ---------------------------------------------------------
        # 종가 및 통화 정보 수집
        # ---------------------------------------------------------
        try:
            data = yf.download(ticker_name, period="1d", interval="1d")
            if not data.empty:
                self.stockdata.last_price = float(data["Close"].iloc[-1])
        except Exception as e:
            print(f"[WARN] yfinance 오류 발생 (가격): {e}")

        try:
            self.stockdata.currency = yf.Ticker(ticker_name).info.get("currency", "USD")
        except Exception as e:
            print(f"[WARN] yfinance 오류 발생 (통화): {e}")
            self.stockdata.currency = "USD"

        print(f"✅ StockData 생성 완료: {ticker_name} / {self.stockdata.currency}")

        return X_tensor, self.stockdata



    # 모델 생성
    def make_lstm_macro_model(self, ticker_name, agent_id, X_train, y_train, scaler_X, scaler_y):
        # -------------------------------------------------------------
        # 11. 단일 아웃풋 LSTM 모델 정의
        # -------------------------------------------------------------
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # 단일 종목 예측
        ])

        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='mae')

        # -------------------------------------------------------------
        # 12. 학습
        # -------------------------------------------------------------
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=60,
            batch_size=16,
            verbose=1
        )


        # 전체 모델 저장
        self.model.save(f"{model_dir}/{ticker_name}_{agent_id}.h5")
        joblib.dump(scaler_X, f"{model_dir}/scaler_X.pkl")
        joblib.dump(scaler_y, f"{model_dir}/scaler_y.pkl")
        print(f"✅ {agent_id} model saved.\n✅ pretraining finished.\n")


    #predict
    def macro_predictor(self, X_seq):
        print("[INFO] 예측 수행 중...")

        # 1. 모델 예측
        pred_scaled = self.model.predict(X_seq)
        pred_inv = self.scaler_y.inverse_transform(pred_scaled)

        # 2. 종가 추출
        last_prices = {}
        for t in self.tickers:
            close_candidates = [c for c in self.macro_df.columns
                                if c.startswith(t) and not c.endswith("_ma5") and "ret" not in c]
            if not close_candidates:
                raise ValueError(f"{t}의 종가 컬럼을 찾을 수 없습니다.")
            last_prices[t] = self.macro_df[close_candidates[0]].iloc[-1]

        # 3. 예측 종가 및 수익률 계산
        records = []
        pred_prices = {}
        for i, t in enumerate(self.tickers):
            pred_ret = float(pred_inv[0][i])
            last_price = float(last_prices[t])
            next_price = last_price * (1 + pred_ret)
            pred_prices[t] = next_price

            records.append({
                "Ticker": t,
                "Last_Close": last_price,
                "Predicted_Close": next_price,
                "Predicted_Return": pred_ret,
                "Predicted_%": pred_ret * 100
            })

            print(f"{t}: 마지막 종가={last_price:.2f} → 예측 종가={next_price:.2f} (예상 수익률 {pred_ret*100:.2f}%)")

        # 4. Monte Carlo Dropout 불확실성
        mean_pred, std_pred, confidence, predicted_price = get_std_pred(
            self.model, X_seq, n_samples=30, scaler_y=self.scaler_y, stockdata=self.stockdata
        )

        # 5. 결과 병합
        for i, r in enumerate(records):
            r["uncertainty"] = float(std_pred[i]) if len(std_pred) > 1 else float(std_pred[-1])
            r["confidence"] = float(confidence[i]) if len(confidence) > 1 else float(confidence[-1])

        pred_df = pd.DataFrame(records).round(4)
        self.pred_df = pred_df
        self.pred_prices = pred_prices

        print("\n================= 예측 결과 (표) =================")
        print(pred_df)

        print("\n================= 예측 결과 (값) =================")
        print(pred_prices)

        # 단일 티커일 경우 target 요약 제공
        self.target = Target(
            next_close=float(pred_df["Predicted_Close"].iloc[-1]),
            uncertainty=float(std_pred[-1]),
            confidence=float(pred_df["confidence"].iloc[-1])
        )


        return self.pred_prices, self.target



    def macro_reviewer_draft(self):
        temporal_summary, causal_summary, interaction_summary = self.make_macro_shap()
        # -------------------------------
        # 4️⃣ llm 생성
        # -------------------------------
        print("\n4️⃣ Generating explanation using LLM...")

        llm  = LLMExplainer()
        feature_summary = feature_df.tail(5).describe().round(3).to_dict()
        explanation = llm.generate_explanation(feature_summary, self.pred_prices,
                                               importance_dict,
                                               temporal_summary, causal_summary,
                                               interaction_summary)

        print(f"\n================= pred_prices:{self.pred_prices} =================")

        print("\n================= LLM Explanation =================")
        print(explanation)
        print("===================================================")

        total_json = {
            'agent_id' : self.agent_id,
            'target' : self.target,
            'reason' : explanation
        }

        stock_data = {
            'temporal_summary' : temporal_summary,
            'causal_summary' : causal_summary,
            'interaction_summary' : interaction_summary

        }

        context = json.dumps({
            "agent_id": self.agent_id,
            "predicted_next_close": round(self.target.next_close, 3),
            "uncertainty_sigma": round(self.target.uncertainty or 0.0, 4),
            "confidence_beta": round(self.target.confidence or 0.0, 4),
            "latest_data": str(stock_data)
        }, ensure_ascii=False, indent=2)

        reason = explanation

        # 4) Opinion 기록/반환 (항상 최신 값 append)
        self.opinions.append(Opinion(agent_id=self.agent_id, target=self.target, reason=reason))

        return total_json, self.opinions[-1]




    def make_macro_shap(self):
        # -------------------------------
        # 3️⃣ SHAP 계산
        # -------------------------------
        # --- (run() 안의 안전 처리) ---
        X_scaled = self.X_scaled.astype(np.float32)
        X_scaled = X_scaled[:, :, :300]
        feature_names = feature_names[:300]

        print("\n3️⃣ Calculating feature importance...")
        analyzer = AttributionAnalyzer(self.model)
        importance_dict, temporal_df, causal_df, interaction_df = analyzer.run_all_shap(X_scaled, feature_names)

        temporal_summary = temporal_df.head().to_dict(orient="records") if temporal_df is not None else []
        causal_summary = causal_df.to_dict(orient="records") if causal_df is not None else []
        if isinstance(interaction_df, pd.DataFrame):
            interaction_summary = interaction_df.iloc[:5, :5].round(3).to_dict()
        else:
            interaction_summary = {}

        return temporal_summary, causal_summary, interaction_summary



    #[base_agent.py]
    def _msg(self, role: str, content: str) -> dict:
        """OpenAI ChatCompletion용 메시지 구조 생성"""
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"_msg() 인자 오류: role={role}, content={type(content)}")
        return {"role": role, "content": content}


    #[base_agent.py]
    def macro_reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성"""

        # 메시지 생성 (context 구성은 별도 헬퍼에서)
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )

        # LLM 호출
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"}
                },
                "required": ["stance", "message"],
                "additionalProperties": False
            }
        )

        # 결과 정리 및 기록
        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        )

        # 저장
        self.rebuttals[round].append(result)

        # 디버깅 로그
        if self.verbose:
            print(
                f"[{self.agent_id}] rebuttal 생성 → {result.stance} "
                f"({my_opinion.agent_id} → {other_opinion.agent_id})"
            )

        return result



    #[m_agent.py]
    def _build_messages_rebuttal(self,
                                 my_opinion: Opinion,
                                 target_opinion: Opinion,
                                 stock_data: StockData) -> tuple[str, str]:

        t = stock_data.ticker or "UNKNOWN"
        ccy = (stock_data.currency or "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        ctx = {
            "ticker": t,
            "currency": ccy,
            "data_summary": getattr(stock_data, self.agent_id, {}).get("feature_cols", []),
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": str(my_opinion.reason)[:2000],
                "uncertainty": float(my_opinion.target.uncertainty),
                "confidence": float(my_opinion.target.confidence),
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "next_close": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:2000],
                "uncertainty": float(target_opinion.target.uncertainty),
                "confidence": float(target_opinion.target.confidence),
            }
        }
        # 각 컬럼별 최근 시계열 그대로 포함
        # (최근 7~14일 정도면 LLM이 이해 가능한 범위)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[self.window_size:]  # 최근 14일치 전체 시계열
            else:
                ctx[col] = [values]

        system_text = REBUTTAL_PROMPTS[self.agent_id]["system"]
        user_text   = REBUTTAL_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text



    #[base_agent.py] OpenAI API 호출
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """모델 폴백 포함 OpenAI Responses API 호출"""
        payload_base = {
            "input": [msg_sys, msg_user],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Response",
                    "strict": True,
                    "schema": schema_obj,
                }
            },
            "temperature": self.temperature,
        }
        last_err = None
        for model in self.preferred_models:
            payload = dict(payload_base, model=model)
            try:
                r = requests.post(self.OPENAI_URL, headers=self.headers, json=payload, timeout=120)
                if r.ok:
                    data = r.json()
                    # 1) output_text 우선 사용
                    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                        try:
                            return json.loads(data["output_text"])
                        except Exception:
                            return {"reason": data["output_text"]}  # JSON 실패 시 원문 텍스트 보존
                    # 2) output 배열에서 텍스트 모으기
                    out = data.get("output")
                    if isinstance(out, list) and out:
                        texts = []
                        for blk in out:
                            for c in blk.get("content", []):
                                if "text" in c:
                                    texts.append(c["text"])
                        joined = "\n".join(t for t in texts if t)
                        if joined.strip():
                            try:
                                return json.loads(joined)
                            except Exception:
                                return {"reason": joined}
                    # 비정상 응답
                    return {}
                # 400/404는 다음 모델로 폴백
                if r.status_code in (400, 404):
                    last_err = (r.status_code, r.text)
                    continue
                # 기타 에러는 즉시 예외
                r.raise_for_status()
            except Exception as e:
                self._p(f"■ 모델 {model} 실패: {e}")
                last_err = str(e)
                continue
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")













    def macro_reviewer_revise(
            self,
            my_opinion: Opinion,
            others: List[Opinion],
            rebuttals: List[Rebuttal],
            stock_data: StockData,
            fine_tune: bool = True,
            lr: float = 1e-4,
            epochs: int = 20,
    ):
        """
        Revision 단계
        - σ 기반 β-weighted 신뢰도 계산
        - γ 수렴율로 예측값 보정
        - fine-tuning (수익률 단위)
        - reasoning 생성
        """
        gamma = getattr(self, "gamma", 0.3)               # 수렴율 (0~1)
        delta_limit = getattr(self, "delta_limit", 0.05)  # fine-tuning 보정 한계

        try:
            # ===================================
            # ① β 계산 (불확실성 작을수록 신뢰 높음)
            # ===================================
            my_price = my_opinion.target.next_close
            my_sigma = abs(my_opinion.target.uncertainty or 1e-6)

            other_prices = np.array([o.target.next_close for o in others])
            other_sigmas = np.array([abs(o.target.uncertainty or 1e-6) for o in others])

            all_sigmas = np.concatenate([[my_sigma], other_sigmas])
            all_prices = np.concatenate([[my_price], other_prices])

            inv_sigmas = 1 / (all_sigmas + 1e-6)
            betas = inv_sigmas / inv_sigmas.sum()

            # ===================================
            # ② 논문식 수렴 업데이트
            #     y_i_rev = y_i + γ Σ β_j (y_j - y_i)
            # ===================================
            delta = np.sum(betas[1:] * (other_prices - my_price))
            revised_price = my_price + gamma * delta

        except Exception as e:
            print(f"[{self.agent_id}] revised_target 계산 실패: {e}")
            revised_price = my_opinion.target.next_close
            current_price = getattr(self.stockdata, "last_price", 100.0)
            price_uplimit = current_price * (1 + delta_limit)
            price_downlimit = current_price * (1 - delta_limit)
            revised_price = min(max(revised_price, price_downlimit), price_uplimit)

        # ===================================
        # ③ Fine-tuning (return 단위)
        # ===================================
        loss_value = None
        if fine_tune and hasattr(self, "model"):
            try:
                current_price = getattr(self.stockdata, "last_price", 100.0)
                revised_return = (revised_price / current_price) - 1  # 🔹수익률 변환

                X_input = self.searcher(self.ticker)
                device = next(self.model.parameters()).device
                X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
                y_tensor = torch.tensor([[revised_return]], dtype=torch.float32).to(device)

                self.model.train()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()

                for _ in range(epochs):
                    optimizer.zero_grad()
                    pred = self.model(X_tensor)
                    delta_loss = pred - y_tensor
                    loss = criterion(pred - delta_loss, y_tensor)
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.item())
                print(f"[{self.agent_id}] fine-tuning 완료: loss={loss_value:.6f}")

            except Exception as e:
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")

        # ===================================
        # ④ fine-tuning 이후 새 예측 생성
        # ===================================
        try:
            X_latest = self.searcher(self.ticker)
            new_target = self.predict(X_latest)
        except Exception as e:
            print(f"[{self.agent_id}] predict 실패: {e}")
            new_target = my_opinion.target

        # ===================================
        # ⑤ reasoning 생성
        # ===================================
        try:
            sys_text, user_text = self._build_messages_revision(
                my_opinion=my_opinion,
                others=others,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )
        except Exception as e:
            print(f"[{self.agent_id}] _build_messages_revision 실패: {e}")
            sys_text, user_text = (
                "너는 금융 분석가다. 간단히 reason만 생성하라.",
                json.dumps({"reason": "기본 메시지 생성 실패"}),
            )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        )

        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=new_target,
            reason=revised_reason,
        )

        self.opinions.append(revised_opinion)
        print(f"[{self.agent_id}] revise 완료 → new_close={new_target.next_close:.2f}, loss={loss_value}")
        return self.opinions[-1]