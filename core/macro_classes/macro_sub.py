import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

from config.agents import dir_info

'''
예측을 위해 최신 매크로 데이터 수집하는 클래스
몬테 카를로 생성 함수도 존재함
'''

# yfinance 진행률 바 및 경고 메시지 숨기기
warnings.filterwarnings("ignore")

model_dir: str = dir_info["model_dir"]


import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MakeDatasetMacro:
    """
    통합 매크로 데이터 생성 클래스
    - 기존 MacroAData, MacroData, macro_sub 기능을 완전 통합
    - 단일 티커(target_ticker) 예측 구조
    """

    def __init__(self, ticker: str, window: int = 40, model_dir: str = None):
        self.ticker = ticker
        self.window = window
        self.model_dir = model_dir or dir_info["model_dir"]
        self.scaler_X = None
        self.scaler_y = None
        self.agent_id = "MacroAgent"
        self.data = None

        self.macro_tickers = [
            "SPY", "QQQ", "^GSPC", "^DJI", "^IXIC",
            "^TNX", "^IRX", "^VIX", "DX-Y.NYB", "EURUSD=X",
            "USDJPY=X", "GC=F", "CL=F", "HG=F"
        ]
        self.x_scaler_path = f"{self.model_dir}/scalers/{ticker}_{self.agent_id}_xscaler.pkl"
        self.y_scaler_path = f"{self.model_dir}/scalers/{ticker}_{self.agent_id}_yscaler.pkl"

        if os.path.exists(self.x_scaler_path):
            self.scaler_X = joblib.load(self.x_scaler_path)
        if os.path.exists(self.y_scaler_path):
            self.scaler_y = joblib.load(self.y_scaler_path)

    # -------------------------------------------------------------
    # 1. 매크로 + 개별 티커 데이터 수집
    # -------------------------------------------------------------
    def fetch_data(self):
        print(f"[INFO] Fetching macro & {self.ticker} data...")

        start_date = datetime.now() - timedelta(days=5*365)
        end_date = datetime.now()

        # (1) 매크로 데이터 다운로드
        df_macro = yf.download(
            tickers=self.macro_tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False
        )

        # ✅ MultiIndex 해제
        if isinstance(df_macro.columns, pd.MultiIndex):
            # ('Close', 'SPY') → 'SPY_Close'
            df_macro.columns = [
                f"{c}_{t}" if isinstance(t, str) else str(t)
                for t, c in df_macro.columns
            ]
        else:
            # fallback: 단일 인덱스면 그대로
            df_macro.columns = [str(c) for c in df_macro.columns]

        # ✅ Close 컬럼만 필터링
        close_cols = [c for c in df_macro.columns if c.endswith("_Close")]
        df_macro = df_macro[close_cols].copy()

        # (2) 개별 티커 데이터 다운로드
        df_stock = yf.download(
            self.ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if isinstance(df_stock, pd.Series):
            df_stock = df_stock.to_frame(name=self.ticker)
        elif "Close" in df_stock.columns:
            df_stock = df_stock[["Close"]].rename(columns={"Close": self.ticker})
        else:
            raise ValueError(f"{self.ticker} 데이터에서 'Close' 컬럼을 찾을 수 없습니다.")

        # ✅ 인덱스 변환
        df_stock.index = pd.to_datetime(df_stock.index)
        df_macro.index = pd.to_datetime(df_macro.index)

        # ✅ 병합
        df = pd.concat([df_stock, df_macro], axis=1)
        df = df.dropna().fillna(method="ffill").fillna(method="bfill")

        # ✅ 검증
        if self.ticker not in df.columns:
            raise KeyError(f"[ERROR] '{self.ticker}' 컬럼이 병합 결과에 없습니다. df.columns={df.columns.tolist()}")

        self.data = df
        return df





    # -------------------------------------------------------------
    # 2. 피처 엔지니어링
    # -------------------------------------------------------------
    def add_features(self):
        df = self.data.copy()

        # 수익률, 스프레드, 리스크 피처
        df["ret_1d"] = df[self.ticker].pct_change()
        if "^TNX" in df.columns and "^IRX" in df.columns:
            df["yield_spread"] = df["^TNX"] - df["^IRX"]
        if "SPY" in df.columns and "DX-Y.NYB" in df.columns and "^VIX" in df.columns:
            df["risk_sentiment"] = df["SPY"] - df["DX-Y.NYB"] - df["^VIX"]

        df = df.dropna().fillna(method="ffill").fillna(method="bfill")

        self.data = df
        return df

    # -------------------------------------------------------------
    # 3. 학습 데이터셋 생성
    # -------------------------------------------------------------
    def build_trainset(self):
        df = self.data.copy()

        # 1️⃣ feature 목록 추출
        features = [c for c in df.columns if c != self.ticker]

        # 2️⃣ feature 이름 평탄화 (멀티인덱스 → 문자열)
        clean_features = []
        for f in features:
            if isinstance(f, tuple):
                # ('NVDA', 'ret_1d') → 'NVDA_ret_1d'
                clean_features.append("_".join([str(x) for x in f if x]))
            else:
                clean_features.append(str(f))

        # 3️⃣ 수익률(target)
        y = df[self.ticker].pct_change().shift(-1)
        y = y.dropna()
        df = df.iloc[1:]

        X = df[features].values
        y = y.values.reshape(-1, 1)

        # ✅ 시퀀스 형태로 변환
        X_seq, y_seq = [], []
        for i in range(len(X) - self.window):
            X_seq.append(X[i:i + self.window])
            y_seq.append(y[i + self.window])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # ✅ DataFrame 기반 스케일링 (feature_names_in_ 유지)
        feature_df = pd.DataFrame(X.reshape(-1, X.shape[-1]), columns=clean_features)

        # ✅ 스케일러 학습
        self.scaler_X = StandardScaler().fit(feature_df)
        self.scaler_y = MinMaxScaler().fit(y)

        # ✅ 스케일러 저장
        os.makedirs(f"{self.model_dir}/scalers", exist_ok=True)
        joblib.dump(self.scaler_X, f"{self.model_dir}/scalers/{self.ticker}_{self.agent_id}_xscaler.pkl")
        joblib.dump(self.scaler_y, f"{self.model_dir}/scalers/{self.ticker}_{self.agent_id}_yscaler.pkl")

        # ✅ 스케일 적용 시에도 동일한 clean_features 사용
        X_scaled = np.array([
            self.scaler_X.transform(pd.DataFrame(seq, columns=clean_features)) for seq in X_seq
        ])
        y_scaled = self.scaler_y.transform(y_seq)

        return X_scaled, y_scaled, clean_features



    # -------------------------------------------------------------
    # 4. 예측용 데이터셋 생성
    # -------------------------------------------------------------
    def build_predictset(self):
        df = self.data.copy()
        features = [c for c in df.columns if c != self.ticker]

        X = df[features].values[-self.window:]

        # ✅ 스케일러 로드
        if self.scaler_X is None:
            if not os.path.exists(self.x_scaler_path):
                raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {self.x_scaler_path}")
            self.scaler_X = joblib.load(self.x_scaler_path)

        if self.scaler_y is None and os.path.exists(self.y_scaler_path):
            self.scaler_y = joblib.load(self.y_scaler_path)

        # ✅ 스케일링 및 리쉐이프
        X_scaled = self.scaler_X.transform(X)
        X_scaled = X_scaled.reshape(1, self.window, -1)

        print(f"[OK] Predictset 생성 완료: X={X_scaled.shape}")
        return X_scaled, features





# -------------------------------------------------------------
# 4. Monte Carlo Dropout
# -------------------------------------------------------------
def get_std_pred(model, X_seq, n_samples=30, scaler_y=None, current_price=None, stockdata=None):
    """
    Monte Carlo Dropout 기반 불확실성 예측 (PyTorch 전용)
    --------------------------------------------------
    기능:
    - Dropout을 training=True로 활성화하여 Monte Carlo 추론 수행
    - n회 반복 예측으로 평균(mean) / 표준편차(std) 계산
    - σ 기반 confidence 계산
    - scaler_y 역변환 지원
    - 현재가 반영한 예측 종가(predicted_price) 계산
    --------------------------------------------------
    Args:
        model : torch.nn.Module (PyTorch 모델)
        X_seq : 입력 시퀀스 (torch.Tensor 또는 numpy.ndarray)
        n_samples : Monte Carlo 반복 횟수
        scaler_y : 출력 스케일러 (MinMaxScaler 또는 StandardScaler)
        current_price : 현재 종가 (float, optional)
        stockdata : StockData 객체 (last_price 가져올 수 있음)
    Returns:
        mean_pred (np.ndarray) : 평균 예측값
        std_pred (np.ndarray)  : 표준편차 (불확실성)
        confidence (float)     : σ 기반 신뢰도 (0~1)
        predicted_price (float): 현재가 × (1 + 예측 수익률)
    """
    import numpy as np
    import torch

    # 입력을 torch.Tensor로 변환
    if isinstance(X_seq, np.ndarray):
        device = next(model.parameters()).device
        X_seq = torch.FloatTensor(X_seq).to(device)
    elif not isinstance(X_seq, torch.Tensor):
        raise TypeError(f"X_seq must be numpy.ndarray or torch.Tensor, got {type(X_seq)}")

    preds = []

    # -------------------------------------------------
    # (1) 모델 예측 (Monte Carlo Dropout)
    # -------------------------------------------------
    model.train()  # training 모드로 설정하여 dropout 활성화
    with torch.no_grad():
        for _ in range(n_samples):
            y_pred = model(X_seq)
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy().flatten()
            preds.append(y_pred)
    model.eval()  # 다시 eval 모드로

    preds = np.stack(preds)  # (n_samples, output_dim)
    mean_pred = preds.mean(axis=0)
    std_pred = np.abs(preds.std(axis=0))

    # -------------------------------------------------
    # (2) σ 기반 confidence 계산
    # -------------------------------------------------
    sigma = float(std_pred[-1]) if std_pred.ndim > 0 else float(std_pred)
    sigma = max(sigma, 1e-6)
    confidence = 1 / (1 + np.log1p(sigma))  # σ 작을수록 1에 가까움

    # -------------------------------------------------
    # (3) 스케일러 역변환
    # -------------------------------------------------
    if scaler_y is not None:
        try:
            mean_pred = scaler_y.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
            std_pred = scaler_y.inverse_transform(std_pred.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"[WARN] scaler_y inverse_transform 실패: {e}")

    # -------------------------------------------------
    # (4) 현재가 및 예측 가격 변환
    # -------------------------------------------------
    if current_price is None:
        if stockdata is not None and hasattr(stockdata, "last_price"):
            current_price = float(getattr(stockdata, "last_price"))
        else:
            current_price = 100.0  # 기본값

    # ✅ 스케일 복원 추가 (scaler_y 없을 때도 기본 복원)
    if scaler_y is None:
        # 모델 출력이 수익률 예측일 경우 → 그대로 사용
        predicted_return = float(mean_pred[-1])
    else:
        # 역변환 적용
        try:
            restored = scaler_y.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
            predicted_return = float(restored[-1])
            mean_pred = restored
        except Exception as e:
            print(f"[WARN] scaler_y inverse_transform 실패: {e}")
            predicted_return = float(mean_pred[-1])

    predicted_price = current_price * (1 + predicted_return)

    # -------------------------------------------------
    # (5) 결과 반환
    # -------------------------------------------------
    return mean_pred, std_pred, confidence, predicted_price


