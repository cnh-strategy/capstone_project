# agents\sentimental_agent.py
from core.sentimental_classes.lstm import SimpleLSTM
from core.sentimental_classes import utils as sutils
from core.sentimental_classes import builders as ctxb
from core.sentimental_classes.price import _ensure_ctx_features_from_window as compute_price_window_stats

class SentimentalAgent(BaseAgent):
    def load_model(self):
        input_dim = self.input_dim or self.infer_input_dim()
        self.model = SimpleLSTM(input_dim, self.hidden_dim or 64,
                                self.num_layers or 1, self.dropout or 0.2)
        self.try_load_state_dict(self.model, self.model_path(self.ticker))
        self.model.eval()

    def build_features(self, raw_df):
        feat_df = sutils.build_price_sentiment_features(raw_df)
        X, y = sutils.build_sequences(feat_df, window=self.window_size)
        return self.scale_features(X), y

    def _enrich_ctx_before_prompt(self, stock_data, target):
        if any(getattr(stock_data, k, None) is None for k in ["lag_ret_1","rolling_vol_20","zscore_close_20"]):
            extra = compute_price_window_stats(self._last_X, getattr(stock_data, "feature_cols", []))
            stock_data.__dict__.update(extra)

    def _build_messages_opinion(self, stock_data, target):
        self._enrich_ctx_before_prompt(stock_data, target)
        ctx = {
          "snapshot": {...},
          "prediction": ctxb.build_prediction_block(stock_data, target),
          "price_features": ctxb.build_price_block(stock_data),
          "volume_features": ctxb.build_volume_block(stock_data),
          "news_features": ctxb.build_news_block(stock_data),
          "regime_features": ctxb.build_regime_block(stock_data),
          "explainability": ctxb.build_explainability_block(stock_data, target),
          "explain_helpers": ctxb.build_explain_helpers(stock_data, target),
        }
        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(
            context=json.dumps(ctx, ensure_ascii=False),
            schema=prompt_set.get("schema", "{}")
        )
        return system_text, user_text