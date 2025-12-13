# ===============================================================
# BaseAgent: LLM 기반 공통 인터페이스
# ===============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple, Any
from dataclasses import field
from collections import defaultdict
import os, json, time, requests, yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from config.agents import agents_info, dir_info, common_params
from core.data_set import build_dataset, load_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib

# ===============================================================
# 데이터 구조 정의
# ===============================================================

@dataclass
class Target:
    """
    예측 목표값 및 불확실성 정보를 담는 데이터 클래스
    
    Attributes:
        next_close (float): 다음 거래일의 예측 종가
        uncertainty (Optional[float]): 예측의 불확실성 (Monte Carlo Dropout 표준편차 σ)
        confidence (Optional[float]): 모델의 신뢰도 β (0~1 사이 값, 불확실성과 반비례)
        predicted_return (Optional[float]): 예측 수익률
    """
    next_close: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    predicted_return: Optional[float] = None
    predicted_return: Optional[float] = None

@dataclass
class Opinion:
    """
    에이전트의 예측 의견을 담는 클래스
    
    Attributes:
        agent_id (str): 의견을 제시한 에이전트 ID
        target (Target): 예측 가격 및 불확실성 정보
        reason (str): 예측의 근거 (LLM이 생성한 텍스트)
    """
    agent_id: str
    target: Target
    reason: str

@dataclass
class Rebuttal:
    """
    타 에이전트 의견에 대한 반박/지지 메시지
    
    Attributes:
        from_agent_id (str): 발신 에이전트 ID
        to_agent_id (str): 수신 에이전트 ID
        stance (Literal["REBUT", "SUPPORT"]): 반박(REBUT) 또는 지지(SUPPORT) 입장
        message (str): 반박 또는 지지의 상세 내용
    """
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str

@dataclass
class RoundLog:
    """
    토론 라운드별 로그 데이터
    
    Attributes:
        round_no (int): 라운드 번호
        opinions (List[Opinion]): 해당 라운드의 에이전트별 의견 목록
        rebuttals (List[Rebuttal]): 해당 라운드의 반박 메시지 목록
        summary (Dict[str, Target]): 라운드 요약 정보
    """
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    """
    에이전트가 사용하는 주식 데이터 컨테이너
    
    Attributes:
        SentimentalAgent (Optional[Dict]): 감성 분석 데이터
        MacroAgent (Optional[Dict]): 거시경제 데이터
        TechnicalAgent (Optional[Dict]): 기술적 분석 데이터
        last_price (Optional[float]): 최신 종가
        currency (Optional[str]): 통화 코드 (예: USD)
        ticker (Optional[str]): 종목 코드
        feature_cols (Optional[List[str]]): 피처 컬럼 이름 목록
    """
    SentimentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    MacroAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    TechnicalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    last_price: Optional[float] = None
    currency: Optional[str] = None
    ticker: Optional[str] = None
    feature_cols: Optional[List[str]] = field(default_factory=list)


# ===============================================================
# BaseAgent 클래스
# ===============================================================
class BaseAgent:
    """
    LLM 기반 Multi-Agent Debate 시스템을 위한 기본 에이전트 클래스.
    모든 개별 에이전트(Technical, Macro, Sentimental)는 이 클래스를 상속받아야 합니다.
    
    주요 기능:
    - 데이터 로드 및 전처리 (searcher, scaler)
    - 모델 학습 및 관리 (pretrain, load_model)
    - 예측 및 불확실성 추정 (predict)
    - LLM 기반 의견 생성 및 토론 참여 (reviewer_draft, reviewer_rebut, reviewer_revise)
    """

    OPENAI_URL = "https://api.openai.com/v1/responses"

    def __init__(
            self,
            agent_id: str,
            model: Optional[str] = None,
            preferred_models: Optional[List[str]] = None,
            temperature: Optional[float] = None,
            verbose: bool = False,
            need_training: bool = True,
            data_dir: str = dir_info["data_dir"],
            model_dir: str = dir_info["model_dir"],
            ticker: str=None,
            gamma: Optional[float] = None,
            delta_limit: Optional[float] = None,
    ):
        """
        BaseAgent 초기화
        
        Args:
            agent_id: 에이전트 식별자 (예: "TechnicalAgent")
            model: 사용할 LLM 모델명
            preferred_models: 모델 폴백 우선순위 리스트
            temperature: LLM 생성 온도
            verbose: 디버그 출력 여부
            need_training: 학습 필요 여부
            data_dir: 데이터 저장 경로
            model_dir: 모델 저장 경로
            ticker: 종목 코드
            gamma: 의견 수렴율 (0~1)
            delta_limit: 최대 변화 허용 폭
        """
        load_dotenv()
        self.agent_id = agent_id
        self.model = model
        
        # Config 로드
        self.temperature = temperature if temperature is not None else common_params.get("temperature", 0.2)
        self.verbose = verbose
        self.need_training = need_training
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.ticker = ticker
        
        # Scaler 초기화
        scaler_dir = os.path.join(model_dir, "scalers")
        self.scaler = DataScaler(agent_id, scaler_dir=scaler_dir)
        self.window_size = agents_info[agent_id]["window_size"]
        
        # 모델 우선순위 설정
        self.preferred_models = preferred_models or common_params.get("preferred_models", ["gpt-5-mini", "gpt-4.1-mini"])
        if model:
            self.preferred_models = [model] + [m for m in self.preferred_models if m != model]

        # API 키 설정
        self.api_key = os.getenv("CAPSTONE_OPENAI_API")
        if not self.api_key:
            raise RuntimeError("환경변수 CAPSTONE_OPENAI_API가 설정되지 않았습니다.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 상태값 초기화
        self.stockdata: Optional[StockData] = None
        self.targets: List[Target] = []
        self.opinions: List[Opinion] = []
        self.rebuttals: Dict[int, List[Rebuttal]] = defaultdict(list)

        # 토론 파라미터
        self.gamma = gamma if gamma is not None else agents_info[agent_id].get("gamma", 0.3)
        self.delta_limit = delta_limit if delta_limit is not None else agents_info[agent_id].get("delta_limit", 0.05)

        # JSON Schema 정의 (LLM 응답 구조)
        self.schema_obj_opinion = {
            "type": "object",
            "properties": {
                "next_close": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["next_close", "reason"],
            "additionalProperties": False,
        }
        self.schema_obj_rebuttal = {
            "type": "object",
            "properties": {
                "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                "message": {"type": "string"},
            },
            "required": ["stance", "message"],
            "additionalProperties": False,
        }

    # ===============================================================
    # 백테스팅 지원 메서드 (외부 호출용)
    # ===============================================================
    def set_test_mode(self, mode: bool):
        """백테스팅 모드 활성화/비활성화"""
        self.test_mode = mode

    def set_simulation_date(self, date_str: str):
        """
        백테스팅 시뮬레이션 기준 날짜 설정
        이 날짜를 기준으로 과거 데이터만 로드하게 됩니다.
        """
        self.simulation_date = date_str
        self.test_mode = True  # 날짜가 설정되면 자동으로 테스트 모드로 간주

    def set_training_window(self, start_date: str):
        """
        백테스팅 학습 시작 날짜 설정
        """
        self.training_start_date = start_date

    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """
        데이터를 검색하고 준비하는 메서드. 
        데이터셋이 없으면 빌드하고, 로드하여 최근 데이터를 반환합니다.
        
        Args:
            ticker: 종목 코드
            rebuild: 데이터셋 강제 재생성 여부
            
        Returns:
            torch.Tensor: 모델 입력용 텐서 (1, Window, Feature)
        """
        import yfinance as yf
        import pandas as pd

        agent_id = self.agent_id

        if ticker is None:
            ticker = self.ticker

        self.ticker = ticker

        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")

        # 데이터셋이 없거나 리빌드 요청 시 생성
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {agent_id} dataset not found. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir)

        # CSV 로드
        X, y, feature_cols = load_dataset(ticker, agent_id=agent_id, save_dir=self.data_dir)

        # StockData 초기화 및 데이터 주입
        self.stockdata = StockData()

        # 최근 window 데이터 추출
        X_latest = X[-1:]
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        # DataFrame으로 변환하여 보관
        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(self.stockdata, agent_id, feature_dict)

        # 종가 및 통화 정보 업데이트
        self.stockdata.ticker = ticker
        try:
            data = yf.download(ticker, period="1d", interval="1d", auto_adjust=False, progress=False)
            val = data["Close"].iloc[-1]
            self.stockdata.last_price = float(val.item() if hasattr(val, "item") else val)
        except Exception as e:
            print(f"yfinance 오류 발생 (last_price)")

        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception as e:
            print(f"yfinance 오류 발생, 통화 기본값 사용: {e}")
            self.stockdata.currency = "USD"

        return X_tensor

    def _infer_current_price(self, X, X_arr, explicit_current_price=None) -> float:
        """
        현재가를 추론하는 내부 헬퍼 메서드.
        1. 인자값 -> 2. StockData -> 3. 입력 데이터 마지막 값 순으로 탐색
        """
        # 0) 명시적 인자 우선
        if explicit_current_price is not None:
            return float(explicit_current_price)

        sd = None
        # 1) X가 StockData 객체인 경우
        try:
            from agents.base_agent import StockData as StockDataType
        except Exception:
            StockDataType = object

        if isinstance(X, StockDataType):
            sd = X
        # 2) self.stockdata 확인
        elif hasattr(self, "stockdata"):
            sd = getattr(self, "stockdata", None)

        # 3) StockData 내 정보 확인
        if sd is not None:
            snap = getattr(sd, "snapshot", None) or getattr(sd, "meta", None) or {}
            if isinstance(snap, dict):
                for key in ("last_price", "current_price", "close", "adj_close"):
                    v = snap.get(key)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
            # StockData 필드 직접 확인
            if getattr(sd, "last_price", None) is not None:
                return float(sd.last_price)

        # 4) 입력 배열의 마지막 값으로 추론
        import numpy as np
        if X_arr is not None and np.ndim(X_arr) >= 2:
            last_step = X_arr[-1]
            if np.ndim(last_step) == 2:
                last_step = last_step[-1]
            try:
                return float(last_step[-1]) # 마지막 피처를 종가로 가정 (주의 필요)
            except Exception:
                pass

        # 5) 추론 실패
        raise RuntimeError(
            "[BaseAgent.predict] current_price를 찾을 수 없습니다. "
            "explicit_current_price를 전달하거나 StockData에 last_price를 설정하세요."
        )

    def _calculate_direction_accuracy_confidence(self) -> Optional[float]:
        """
        최근 N일 동안의 방향정확도를 계산하여 신뢰도로 반환
        
        Returns:
            float: 방향정확도 기반 신뢰도 (0~1 범위), 계산 실패시 None
        """
        import pandas as pd
        
        try:
            # 1. config에서 lookback_days 읽기
            lookback_days = common_params.get("confidence_lookback_days", 30)
            
            # 2. 필수 정보 확인
            if not hasattr(self, "ticker") or not self.ticker:
                return None
            if not hasattr(self, "agent_id") or not self.agent_id:
                return None
            if not hasattr(self, "data_dir") or not self.data_dir:
                return None
            
            # 3. dataset.csv 파일 경로 확인
            dataset_path = os.path.join(self.data_dir, f"{self.ticker}_{self.agent_id}_dataset.csv")
            if not os.path.exists(dataset_path):
                return None
            
            # 4. 최근 N개 샘플 로드
            df = pd.read_csv(dataset_path)
            
            # 피처 컬럼 추출 (sample_id, time_step, target, date 제외)
            meta_cols = {"sample_id", "time_step", "target", "date"}
            feature_cols = [
                c for c in df.columns
                if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
            
            if len(feature_cols) == 0:
                return None
            
            unique_samples = sorted(df['sample_id'].unique())
            if len(unique_samples) < lookback_days:
                return None
            
            # 마지막 N개 샘플 선택
            recent_samples = unique_samples[-lookback_days:]
            
            # 5. 모델이 로드되어 있는지 확인
            if not hasattr(self, "model") or self.model is None:
                return None
            
            # 6. 각 샘플에 대해 예측 수행 및 방향 비교
            correct_count = 0
            total_count = 0
            
            # 재귀 방지 플래그 설정
            if not hasattr(self, "_calculating_confidence"):
                self._calculating_confidence = False
            
            if self._calculating_confidence:
                return None  # 재귀 호출 방지
            
            self._calculating_confidence = True
            
            try:
                # Device 설정
                if hasattr(self, "device"):
                    device = self.device
                elif hasattr(self.model, "parameters"):
                    try:
                        device = next(self.model.parameters()).device
                    except StopIteration:
                        device = torch.device("cpu")
                else:
                    device = torch.device("cpu")
                
                self.model.eval()  # 평가 모드로 설정 (Dropout 비활성화)
                
                for sample_id in recent_samples:
                    try:
                        # 샘플 데이터 추출
                        sample_data = df[df['sample_id'] == sample_id].sort_values('time_step')
                        if len(sample_data) == 0:
                            continue
                        
                        # X 데이터 추출 (window_size, n_features)
                        X_sample = sample_data[feature_cols].values.astype(np.float32)
                        y_actual = sample_data['target'].iloc[-1]  # 실제값 (수익률)
                        
                        # NaN 체크
                        if np.isnan(y_actual) or np.any(np.isnan(X_sample)):
                            continue
                        
                        # 텐서 변환
                        X_tensor = torch.from_numpy(X_sample).unsqueeze(0).to(device)  # (1, window_size, n_features)
                        
                        # 모델로 예측 (단일 샘플, Dropout 비활성화)
                        with torch.no_grad():
                            out = self.model(X_tensor)
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                            y_pred = out.detach().cpu().numpy().squeeze()
                        
                        # 스케일러 역변환이 필요한 경우 처리
                        if hasattr(self, "scaler") and hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
                            try:
                                y_pred_scaled = np.array([[y_pred]])
                                y_pred = self.scaler.inverse_y(y_pred_scaled)[0, 0]
                            except Exception:
                                pass
                        
                        # 방향 비교 (수익률이므로 부호 비교)
                        if np.sign(y_pred) == np.sign(y_actual):
                            correct_count += 1
                        total_count += 1
                        
                    except Exception as e:
                        # 개별 샘플 처리 실패 시 스킵
                        continue
                
            finally:
                self._calculating_confidence = False
            
            # 7. 방향정확도 = 맞은 수 / 전체 수
            if total_count == 0:
                return None
            
            direction_accuracy = correct_count / total_count
            return float(direction_accuracy)  # 0~1 범위
            
        except Exception as e:
            # 전체 계산 실패 시 None 반환
            return None

    def predict(self, X, n_samples: Optional[int] = None, current_price: float | None = None):
        """
        Monte Carlo Dropout을 이용한 예측 수행.
        
        Args:
            X: 입력 데이터 (StockData, numpy array, or torch tensor)
            n_samples: MC Dropout 샘플링 횟수
            current_price: 현재가 (수익률 -> 가격 변환용)
            
        Returns:
            Target: 예측된 종가, 불확실성, 신뢰도 포함
        """
        import numpy as np
        import torch
        
        if n_samples is None:
            n_samples = common_params.get("n_samples", 30)

        X_original = X

        # 0) StockData 처리
        try:
            from agents.base_agent import StockData as _StockData
        except Exception:
            _StockData = None

        if _StockData is not None and isinstance(X, _StockData):
            # StockData에서 배열 추출 시도
            for name in ["X", "x", "X_seq", "data", "inputs"]:
                if hasattr(X, name):
                    X_arr = getattr(X, name)
                    break
            else:
                # __dict__ 탐색
                X_arr = None
                if hasattr(X, "__dict__"):
                    for name, val in X.__dict__.items():
                        if isinstance(val, (np.ndarray, torch.Tensor)):
                            X_arr = val
                            break
            
                if X_arr is None:
                    raise AttributeError("StockData에서 입력 배열을 찾을 수 없습니다.")
            X = X_arr

        # 1) 데이터 정규화 및 텐서 변환
        if isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            X_np = np.asarray(X, dtype=np.float32)
            X_tensor = torch.from_numpy(X_np)

        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(0)

        # Device 설정
        if hasattr(self, "device"):
            device = self.device
        elif hasattr(self, "model") and hasattr(self.model, "parameters"):
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        X_tensor = X_tensor.to(device)

        # 모델 로드 확인 (재귀 방지 플래그 확인)
        if not hasattr(self, "_in_pretrain"):
            self._in_pretrain = False
        
        if not hasattr(self, "model") or self.model is None:
             if not self.load_model():
                 # 모델이 없으면 pretrain 시도 (재귀 방지)
                 if not self._in_pretrain:
                     print(f"[{self.agent_id}] 모델이 로드되지 않아 pretrain을 시도합니다.")
                     self._in_pretrain = True
                     try:
                         self.pretrain()
                     finally:
                         self._in_pretrain = False
                 else:
                     raise RuntimeError(f"[{self.agent_id}] pretrain 중 predict 호출로 인한 재귀 호출 방지")

        # 2) Monte Carlo Dropout 실행
        self.model.train() # Dropout 활성화
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.model(X_tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                preds.append(out.detach().cpu().numpy())

        preds_arr = np.stack(preds, axis=0)
        mean_pred = preds_arr.mean(axis=0).squeeze()
        std_pred = preds_arr.std(axis=0).squeeze()

        # 불확실성 계산
        if np.ndim(std_pred) > 0:
            sigma = float(std_pred[-1])
        else:
            sigma = float(std_pred)

        # 신뢰도 계산 (방향정확도 기반 우선, 실패 시 불확실성 기반 fallback)
        confidence = self._calculate_direction_accuracy_confidence()
        if confidence is None:
            # Fallback: 기존 불확실성 기반 계산
            confidence_formula = common_params.get("confidence_formula", "1.0 / (1.0 + sigma)")
            confidence = float(eval(confidence_formula))

        # 3) 현재가 결정
        X_arr_for_price = X_tensor.detach().cpu().numpy()
        current_price_val = self._infer_current_price(
            X_original,
            X_arr_for_price,
            explicit_current_price=current_price,
        )

        # 4) 수익률 -> 종가 변환
        mean_pred = np.asarray(mean_pred)
        if mean_pred.ndim == 0:
            predicted_return = float(mean_pred)
        else:
            predicted_return = float(mean_pred[-1])

        # y_scaler가 있다면 역변환 필요할 수 있음 (Agent 구현에 따라 다름)
        # BaseAgent의 predict는 기본적으로 모델이 raw return을 뱉는다고 가정하거나,
        # 상속받은 클래스에서 처리하도록 함. 여기서는 기본 로직만 제공.
        
        predicted_price = current_price_val * (1.0 + predicted_return)

        # 5) 불확실성 처리
        if std_pred is not None:
            std_pred = np.asarray(std_pred)
            if std_pred.ndim == 0:
                uncertainty = float(std_pred)
            else:
                uncertainty = float(std_pred[-1])
        else:
            uncertainty = None

        target = Target(
            next_close=predicted_price,
            uncertainty=uncertainty,
            confidence=confidence,
        )
        return target

    # -----------------------------
    # 메인 워크플로 메서드
    # -----------------------------
    def reviewer_draft(self, stock_data=None, target=None):
        """
        초기 의견(Opinion)을 생성합니다.
        1. 데이터 수집
        2. 모델 예측 (Target 생성)
        3. LLM을 통한 근거(Reason) 생성
        """
        # 1) 데이터 수집
        if stock_data is None:
            sd = getattr(self, "stockdata", None)
            if sd is None:
                # 데이터가 없으면 searcher 시도
                try:
                    self.searcher()
                    sd = self.stockdata
                except Exception:
                    pass
            
            if sd is None:
                raise RuntimeError(f"[{self.agent_id}] StockData가 없습니다.")
            
            if isinstance(sd, dict):
                stock_data = sd.get(self.agent_id, None)
            else:
                stock_data = sd

        # 2) 예측값 생성
        if target is None:
            target = self.predict(stock_data)

        # 3) LLM 호출 (Reason 생성)
        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}
        )

        reason = parsed.get("reason", "(사유 생성 실패)")

        # 4) 저장 및 반환
        self.opinions.append(Opinion(agent_id=self.agent_id, target=target, reason=reason))
        return self.opinions[-1]

    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """
        상대방 의견에 대한 반박(Rebuttal)을 생성합니다.
        """
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )

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

        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        )

        self.rebuttals[round].append(result)
        if self.verbose:
            print(f"[{self.agent_id}] Rebuttal: {result.stance} -> {other_opinion.agent_id}")

        return result

    def _calculate_consensus_price(self, my_opinion: Opinion, others: List[Opinion]) -> float:
        """
        불확실성(σ) 기반 가중치(β)를 사용하여 합의된 가격을 계산합니다.
        공식: y_new = y_me + γ * Σ [β_other * (y_other - y_me)]
        """
        gamma = getattr(self, "gamma", 0.3)
        try:
            my_price = float(my_opinion.target.next_close)
            sigma_min = common_params.get("sigma_min", 1e-6)
            my_sigma = abs(my_opinion.target.uncertainty or sigma_min)

            if not others:
                return my_price

            other_prices = np.array([o.target.next_close for o in others], dtype=float)
            other_sigmas = np.array([abs(o.target.uncertainty or sigma_min) for o in others], dtype=float)

            # 전체 시그마 취합 (나 + 타인들)
            all_sigmas = np.concatenate([[my_sigma], other_sigmas])
            # 불확실성이 낮을수록 가중치 높음 (역수 비례)
            inv_sigmas = 1 / (all_sigmas + sigma_min)
            betas = inv_sigmas / inv_sigmas.sum()

            # 타인들의 가중치 * 가격차이 합산
            delta = np.sum(betas[1:] * (other_prices - my_price))
            revised_price = my_price + gamma * delta
            return float(revised_price)

        except Exception as e:
            print(f"[{self.agent_id}] _calculate_consensus_price 실패: {e}")
            return float(my_opinion.target.next_close)

    def reviewer_revise(
            self,
            my_opinion: Opinion,
            others: List[Opinion],
            rebuttals: List[Rebuttal],
            stock_data: StockData,
            fine_tune: bool = True,
            lr: Optional[float] = None,
            epochs: Optional[int] = None,
    ):
        """
        토론 후 자신의 예측을 수정(Revise)합니다.
        1. Consensus Price 계산
        2. 모델 Fine-tuning (선택적)
        3. 재예측 및 새로운 Reason 생성
        """
        # 파라미터 설정
        if lr is None:
            lr = common_params.get("fine_tune_lr", 1e-4)
        if epochs is None:
            epochs = agents_info.get(self.agent_id, {}).get("fine_tune_epochs", common_params.get("fine_tune_epochs", 10))
        
        # 1. 합의 가격 계산
        revised_price = self._calculate_consensus_price(my_opinion, others)

        # 2. Fine-tuning 및 재예측을 위한 데이터 준비 (한 번만 호출)
        X_latest = None
        loss_value = None
        model = getattr(self, "model", None)
        if model is None and isinstance(self, torch.nn.Module):
            model = self

        # searcher를 한 번만 호출하여 데이터 준비
        try:
            X_latest = self.searcher(self.ticker)
        except Exception as e:
            print(f"[{self.agent_id}] searcher 호출 실패: {e}")
            # searcher 실패 시 합의 가격 사용
            predicted_target = Target(
                next_close=float(revised_price),
                uncertainty=my_opinion.target.uncertainty,
                confidence=my_opinion.target.confidence
            )
            # LLM Revision 메시지 생성으로 건너뛰기
            try:
                sys_text, user_text = self._build_messages_revision(
                    my_opinion=my_opinion,
                    others=others,
                    rebuttals=rebuttals,
                    stock_data=stock_data,
                )
            except Exception as e:
                print(f"[{self.agent_id}] Revision 메시지 생성 실패: {e}")
                sys_text, user_text = ("금융 분석가입니다.", json.dumps({"reason": "메시지 생성 실패"}))
            
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
                target=predicted_target,
                reason=revised_reason,
            )
            self.opinions.append(revised_opinion)
            return revised_opinion

        if fine_tune and model is not None and X_latest is not None:
            try:
                current_price = getattr(stock_data, "last_price", None)
                if current_price is None:
                    default_price = common_params.get("default_current_price", 100.0)
                    current_price = getattr(self, "last_price", default_price)

                # 목표 수익률 계산
                revised_return = (revised_price / current_price) - 1.0
                
                # 스케일링
                y_scale_factor = common_params.get("y_scale_factor", 100.0)
                revised_return_scaled = revised_return * y_scale_factor
                
                # y_scaler 적용
                if hasattr(self, "scaler") and getattr(self.scaler, "y_scaler", None) is not None:
                    y_target_scaled = self.scaler.y_scaler.transform(
                        np.array([[revised_return_scaled]], dtype=float)
                    )[0, 0]
                else:
                    y_target_scaled = revised_return_scaled

                # 학습 데이터 준비 (이미 X_latest를 가져왔으므로 재사용)
                device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
                
                if isinstance(X_latest, torch.Tensor):
                    X_tensor = X_latest.to(device).float()
                else:
                    X_tensor = torch.tensor(X_latest, dtype=torch.float32).to(device)
                
                y_tensor = torch.tensor([[y_target_scaled]], dtype=torch.float32).to(device)

                # 학습 수행
                model.train()
                try:
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    huber_delta = common_params.get("huber_loss_delta", 1.0)
                    criterion = torch.nn.HuberLoss(delta=huber_delta)

                    for _ in range(epochs):
                        optimizer.zero_grad()
                        pred = model(X_tensor)
                        loss = criterion(pred, y_tensor)
                        loss.backward()
                        optimizer.step()

                    loss_value = float(loss.item())
                    print(f"[{self.agent_id}] Fine-tuning 완료: loss={loss_value:.6f}")
                finally:
                    model.eval()

            except Exception as e:
                print(f"[{self.agent_id}] Fine-tuning 실패: {e}")

        # 3. 재예측 (Target 갱신) - X_latest 재사용
        try:
            predicted_target = self.predict(X_latest, current_price=getattr(stock_data, "last_price", None))
        except Exception as e:
            print(f"[{self.agent_id}] 재예측 실패, 합의 가격 사용: {e}")
            predicted_target = Target(
                next_close=float(revised_price),
                uncertainty=my_opinion.target.uncertainty,
                confidence=my_opinion.target.confidence
            )

        # 4. LLM Revision 메시지 생성
        try:
            sys_text, user_text = self._build_messages_revision(
                my_opinion=my_opinion,
                others=others,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )
        except Exception as e:
            print(f"[{self.agent_id}] Revision 메시지 생성 실패: {e}")
            sys_text, user_text = ("금융 분석가입니다.", json.dumps({"reason": "메시지 생성 실패"}))

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
        
        # 결과 저장
        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=predicted_target,
            reason=revised_reason,
        )
        self.opinions.append(revised_opinion)
        
        return revised_opinion

    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        """Opinion 생성을 위한 LLM 메시지를 구성합니다. (하위 클래스 구현 필요)"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_opinion method")

    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        """Rebuttal 생성을 위한 LLM 메시지를 구성합니다. (하위 클래스 구현 필요)"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_rebuttal method")
    
    def _build_messages_revision(self, *args, **kwargs) -> Tuple[str, str]:
        """Revision 생성을 위한 LLM 메시지를 구성합니다. (하위 클래스 구현 필요)"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_revision method")

    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치를 로드합니다."""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            return False

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

            # 모델 인스턴스 생성 (없을 경우)
            if getattr(self, "model", None) is None:
                if hasattr(self, "_build_model"):
                    self.model = self._build_model()
                elif hasattr(self, "forward"):
                    self.model = self
                else:
                    raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되어 있지 않습니다.")

            model = self.model
            
            # 가중치 로드
            if isinstance(checkpoint, torch.nn.Module):
                model.load_state_dict(checkpoint.state_dict())
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
                model.load_state_dict(state_dict)
            else:
                print(f"알 수 없는 체크포인트 포맷: {type(checkpoint)}")
                return False

            self.model = model
            model.eval()
            self.model_loaded = True
            print(f"[{self.agent_id}] 모델 로드 완료: {model_path}")
            return True

        except Exception as e:
            print(f"[{self.agent_id}] 모델 로드 실패: {e}")
            return False

    def pretrain(self):
        """
        에이전트별 모델을 사전 학습(Pre-training)합니다.
        1. 데이터 로드
        2. 스케일링
        3. 모델 생성 및 학습
        4. 모델 저장
        """
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]

        # 1. 데이터 로드
        X, y, cols = load_dataset(self.ticker, self.agent_id, save_dir=self.data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id} ({len(X)} samples)")

        # 백테스팅 모드 확인
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용")

        # 2. 스케일링
        # 타겟 스케일링 (수익률 * factor)
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_train = y * y_scale_factor
        X_train = X

        self.scaler.fit_scalers(X_train, y_train)
        self.scaler.save(self.ticker)

        X_train, y_train = map(torch.tensor, self.scaler.transform(X_train, y_train))
        X_train, y_train = X_train.float(), y_train.float()

        # 3. 모델 생성
        if getattr(self, "model", None) is None:
            if hasattr(self, "_build_model"):
                self.model = self._build_model()
            else:
                raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되지 않음")

        model = self.model
        model.train()

        # 4. 학습
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        loss_fn_name = agents_info.get(self.agent_id, {}).get("loss_fn", "HuberLoss")
        if loss_fn_name == "HuberLoss":
            huber_delta = common_params.get("huber_loss_delta", 1.0)
            loss_fn = torch.nn.HuberLoss(delta=huber_delta)
        elif loss_fn_name == "L1Loss":
            loss_fn = torch.nn.L1Loss()
        elif loss_fn_name == "MSELoss":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.HuberLoss()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in train_loader:
                y_pred = model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.6f}")

        # 5. 저장
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        self.model_loaded = True
        print(f"[{self.agent_id}] 모델 학습 및 저장 완료: {model_path}")

    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """
        OpenAI API 호출 (Fallback 지원)
        백테스팅 모드일 경우 더미 응답 반환
        """
        # 백테스팅 모드 처리
        if hasattr(self, 'test_mode') and self.test_mode:
            dummy_response = {}
            if schema_obj and isinstance(schema_obj, dict):
                props = schema_obj.get("properties", {})
                for key in props.keys():
                    if key == "reason":
                        dummy_response[key] = f"[백테스팅 모드] {self.agent_id} 예측 근거"
                    elif key == "stance":
                        dummy_response[key] = "SUPPORT"
                    elif key == "message":
                        dummy_response[key] = f"[백테스팅 모드] {self.agent_id} 메시지"
                    else:
                        prop_type = props[key].get("type", "string")
                        if prop_type == "string":
                            dummy_response[key] = ""
                        elif prop_type == "number":
                            dummy_response[key] = 0.0
                        else:
                            dummy_response[key] = None
            else:
                dummy_response = {"reason": f"[백테스팅 모드] {self.agent_id} 응답"}
            return dummy_response

        if not msg_sys or not msg_user:
            raise ValueError("Invalid messages: system or user message is None.")

        # JSON Schema 보완
        if schema_obj and isinstance(schema_obj, dict):
            schema_obj.setdefault("additionalProperties", False)
            if "type" not in schema_obj:
                schema_obj["type"] = "object"

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
                    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                        try:
                            return json.loads(data["output_text"])
                        except Exception:
                            return {"reason": data["output_text"]}
                    
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
                    return {}
                
                if r.status_code in (400, 404):
                    last_err = (r.status_code, r.text)
                    continue
                r.raise_for_status()
            except Exception as e:
                last_err = str(e)
                continue
        
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    def _msg(self, role: str, content: str) -> dict:
        """OpenAI 메시지 포맷 생성"""
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"_msg() 인자 오류: role={role}, content={type(content)}")
        return {"role": role, "content": content}


# ===============================================================
# DataScaler 클래스
# ===============================================================
class DataScaler:
    """
    데이터 정규화(Scaling)를 담당하는 클래스
    """
    def __init__(self, agent_id, scaler_dir: Optional[str] = None):
        self.agent_id = agent_id
        if scaler_dir is None:
            scaler_dir = dir_info.get("scaler_dir", os.path.join(dir_info["model_dir"], "scalers"))
        self.save_dir = scaler_dir
        self.x_scaler = agents_info[self.agent_id]["x_scaler"]
        self.y_scaler = agents_info[self.agent_id]["y_scaler"]

    def fit_scalers(self, X_train, y_train):
        """Scales Fit"""
        ScalerMap = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "None": None,
        }
        Sx = ScalerMap.get(self.x_scaler)
        Sy = ScalerMap.get(self.y_scaler)

        # 3D 입력(samples, seq_len, features) -> 2D 변환 후 fit
        n_samples, seq_len, n_feats = X_train.shape
        X_2d = X_train.reshape(-1, n_feats)
        
        self.x_scaler = Sx().fit(X_2d) if Sx else None
        self.y_scaler = Sy().fit(y_train.reshape(-1, 1)) if Sy else None

    def transform(self, X, y=None):
        """Transform"""
        if X.ndim == 3:
            n_samples, seq_len, n_feats = X.shape
            X_2d = X.reshape(-1, n_feats)
            X_t = self.x_scaler.transform(X_2d).reshape(n_samples, seq_len, n_feats) if self.x_scaler else X
        else:
            X_t = self.x_scaler.transform(X) if self.x_scaler else X

        y_t = y
        if y is not None and self.y_scaler:
            y_t = self.y_scaler.transform(y.reshape(-1, 1)).flatten()
            
        return X_t, y_t

    def inverse_y(self, y_pred):
        """Inverse Transform for Y"""
        if self.y_scaler and self.y_scaler != "None" and hasattr(self.y_scaler, 'inverse_transform'):
            if isinstance(y_pred, (list, tuple)):
                y_pred = np.array(y_pred)
            return self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred

    def save(self, ticker):
        """Scaler 저장"""
        os.makedirs(self.save_dir, exist_ok=True)
        if self.x_scaler:
            joblib.dump(self.x_scaler, os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_xscaler.pkl"))
        if self.y_scaler:
            joblib.dump(self.y_scaler, os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_yscaler.pkl"))

    def load(self, ticker):
        """Scaler 로드"""
        x_path = os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_xscaler.pkl")
        y_path = os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_yscaler.pkl")
        if os.path.exists(x_path):
            self.x_scaler = joblib.load(x_path)
        if os.path.exists(y_path):
            self.y_scaler = joblib.load(y_path)
