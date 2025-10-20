# ===============================================================
# File: fundamental_agent_with_alpha_transformer.py
# Purpose: LLM 기반 수식 알파 생성 + Transformer 기반 주가 예측
# ===============================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime
from openai import OpenAI
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from fundamental_sub import MacroSentimentAgent



# ===============================================================
# 1️⃣ LLM 기반 수식 알파 생성기
# ===============================================================
class AlphaGenerator:
    """
    LLM을 이용해 새로운 수식형 알파(formulaic alpha)를 생성하는 모듈.
    """

    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_alpha_with_explanation(self, feature_list, n_alphas=3):
        """
        LLM이 알파 수식을 생성하고, 각 수식의 논리적 근거를 설명.
        """
        feature_str = ", ".join(feature_list[:30])
        prompt = f"""
        다음 feature들을 이용해 주가 예측용 알파 수식을 {n_alphas}개 만들어주세요.
        각 알파는 '수식'과 '생성 이유'를 함께 포함해야 합니다.

        예시 출력:
        alpha1 = (EMA5 - EMA20) / (RSI + 1)
        설명: 단기-장기 이동평균의 차이를 RSI로 정규화하여 과매수/과매도 구간의 영향을 줄입니다.

        feature 목록:
        {feature_str}

        출력 형식:
        alpha1 = ...
        설명: ...
        alpha2 = ...
        설명: ...
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )

        lines = response.choices[0].message.content.strip().split("\n")
        alphas, explanations = [], []
        current_alpha = None

        for line in lines:
            if line.startswith("alpha"):
                current_alpha = line.split("=")[-1].strip()
                alphas.append(current_alpha)
            elif line.startswith("설명") or line.lower().startswith("explanation"):
                explanations.append(line.split(":")[-1].strip())

        return alphas, explanations


# ===============================================================
# 2️⃣ 알파 수식 계산 모듈
# ===============================================================
class AlphaEvaluator:
    """
    LLM이 생성한 수식을 실제 데이터프레임에 적용하여 알파 값을 계산.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_alphas(self, formulas):
        for i, formula in enumerate(formulas):
            col_name = f"alpha_{i+1}"
            try:
                self.df[col_name] = self.df.eval(formula)
                print(f"[AlphaEvaluator] {col_name} 계산 완료: {formula}")
            except Exception as e:
                print(f"[AlphaEvaluator] {col_name} 계산 실패 ({e})")
                self.df[col_name] = 0
        return self.df


# ===============================================================
# 3️⃣ Transformer 기반 시계열 예측 모델
# ===============================================================
class TransformerPredictor:
    """
    Transformer 기반의 다중 시계열 예측 모델.
    """

    def __init__(self, seq_len=40, d_model=64, num_heads=4, ff_dim=128, output_dim=3):
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.output_dim = output_dim
        self.model = None

    def build_model(self, input_dim):
        inputs = Input(shape=(self.seq_len, input_dim))
        x = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)(inputs, inputs)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(self.ff_dim, activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(self.output_dim)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(1e-3), loss="mse")
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=8):
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs, batch_size=batch_size,
                       callbacks=[es], verbose=1)

    def predict(self, X_input):
        return self.model.predict(X_input)


# ===============================================================
# 4️⃣ 통합 실행 클래스 (AlphaForecastAgent)
# ===============================================================
class AlphaForecastAgent:
    """
    전체 파이프라인:
    1. MacroSentimentAgent로 데이터 수집
    2. LLM으로 알파 수식 생성
    3. 알파 계산 후 feature 통합
    4. Transformer 모델로 예측
    """

    def __init__(self, base_date=datetime.today(), window=40):
        self.base_date = base_date
        self.window = window

    def run(self):
        # Step 1. 매크로 데이터 수집
        print("1️⃣ Fetching macro data...")
        macro_agent = MacroSentimentAgent(base_date=self.base_date, window=self.window)
        macro_agent.fetch_data()
        df = macro_agent.add_features()

        # Step 2. LLM을 통한 알파 수식 생성
        print("2️⃣ Generating formulaic alphas using LLM...")
        alpha_gen = AlphaGenerator()
        formulas, explanations = alpha_gen.generate_alpha_with_explanation(df.columns.tolist(), n_alphas=3)

        print("Generated Alphas:", formulas)

        # Step 3. 수식 평가 및 feature 통합
        print("3️⃣ Evaluating and merging alphas...")
        evaluator = AlphaEvaluator(df)
        df = evaluator.compute_alphas(formulas)
        # Step 3 이후 출력 부분 추가
        print("\n[LLM Reasoning Output]")
        for i, (f, e) in enumerate(zip(formulas, explanations), 1):
            print(f"α{i}: {f}")
            print(f"  이유: {e}")


    # Step 4. Transformer 입력 구성
        print("4️⃣ Preparing sequences for Transformer...")
        feature_cols = [c for c in df.columns if c != "Date"]
        X = df[feature_cols].values
        y = df[["AAPL_ma5", "MSFT_ma5", "NVDA_ma5"]].shift(-1).fillna(method="ffill").values

        X_seq, y_seq = [], []
        for i in range(len(X) - self.window):
            X_seq.append(X[i:i+self.window])
            y_seq.append(y[i+self.window-1])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        split_idx = int(len(X_seq) * 0.8)
        X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
        X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]

        # Step 5. Transformer 모델 훈련
        print("5️⃣ Training Transformer model...")
        transformer = TransformerPredictor(seq_len=self.window)
        model = transformer.build_model(input_dim=X_seq.shape[2])
        transformer.train(X_train, y_train, X_val, y_val)

        # Step 6. 최근 시퀀스 예측
        print("6️⃣ Predicting next-day prices...")
        X_latest = X_seq[-1:]
        y_pred = transformer.predict(X_latest)[0]
        tickers = ["AAPL", "MSFT", "NVDA"]
        pred_df = pd.DataFrame({"Ticker": tickers, "Predicted_Close": y_pred})
        print(pred_df)

        return pred_df, formulas


# ===============================================================
# 실행 예시
# ===============================================================
if __name__ == "__main__":
    agent = AlphaForecastAgent()
    predictions, generated_formulas = agent.run()
    print("\n================= Generated Alpha Formulas =================")
    for i, f in enumerate(generated_formulas, 1):
        print(f"α{i}:", f)
    print("\n================= Predicted Prices =================")
    print(predictions.round(3))
