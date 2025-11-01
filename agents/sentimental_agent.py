def build_or_load_model(self, input_dim: int) -> nn.Module:
    if self.model is None:
        self.model = _SentimentalNet(
            input_dim=input_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout
        )
        self.model.to(self.device)
        # 필요 시 self.load_weights(...) 호출 가능
    return self.model


# -------------------------------------
# 3) 예측 수행
# -------------------------------------
def predict_next(self, X_last: np.ndarray) -> Dict[str, Any]:
    self.model.eval()
    with torch.no_grad():
        x = torch.tensor(X_last[None, ...], dtype=torch.float32, device=self.device)
        pred_r = self.model(x).item()  # 다음날 수익률

        # 가격 변환 (BaseAgent 제공 가정)
        last_close = float(self.last_close())
        pred_close = last_close * (1.0 + pred_r)

        # 간단한 불확실성(표준편차) 추정: 드롭아웃 MC 또는 과거 오차 기반 등 실제 구현에 맞게 대체
        uncertainty = float(abs(pred_r))  # placeholder
        confidence = float(np.clip(1.0 - abs(pred_r), 0.0, 1.0))

    return {
        "pred_return": pred_r,
        "pred_close": pred_close,
        "uncertainty": uncertainty,
        "confidence": confidence,
    }


# -------------------------------------
# 4) 사용자 중심 이유 텍스트 상위 4개 만들기
# -------------------------------------
def make_user_reasons(self, df) -> List[Tuple[str, float]]:
    # 예: 최근 윈도우의 피처 평균/변화 등을 기반으로 사용자 친화 라벨로 변환
    label_map = user_reason_labels()
    return pick_top_reason_texts(df.tail(self.cfg.window), label_map)


# -------------------------------------
# 5) 통합 실행 (의견 생성)
# -------------------------------------
def opinion(self) -> Dict[str, Any]:
    data = self.load_data()
    X, y, df = data["X"], data["y"], data["df"]

    # 스케일러: BaseAgent 공용 함수 사용
    scaler = self.fit_or_load_scaler(X)
    X_scaled = self.apply_scaler(scaler, X)

    model = self.build_or_load_model(input_dim=X_scaled.shape[-1])
    pred = self.predict_next(X_scaled[-1])
    reasons = self.make_user_reasons(df)

    text = render_yj_style(
        ticker=self.ticker,
        pred_return=pred["pred_return"],
        pred_close=pred["pred_close"],
        reasons_ranked=reasons,
        confidence=pred["confidence"],
        uncertainty=pred["uncertainty"],
    )

    return {
        "agent_id": self.agent_id,
        "ticker": self.ticker,
        "prediction": pred,
        "text": text,
    }


# debate 진입점 (필요하면..)
def run_debate(self):
    return self.opinion()