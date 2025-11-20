# -*- coding: utf-8 -*-
"""
MacroAnalysisReviewer - 최신 안정판
MacroAgent를 predictor로 받아서:
1) reviewer_draft
2) reviewer_rebut
3) reviewer_revise
를 완전 구현.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import asdict

from agents.base_agent import StockData, Target, Opinion, Rebuttal
from core.macro_classes.gradient_analyzer import GradientAnalyzer
from core.macro_classes.macro_llm import LLMExplainer
from prompts.agent_prompts import (
    OPINION_PROMPTS,
    REBUTTAL_PROMPTS,
    REVISION_PROMPTS,
)


class MacroAnalysisReviewer:

    # --------------------------------------------------------------
    # 초기화: MacroAgent(predictor)를 통째로 넘김
    # --------------------------------------------------------------
    def __init__(self, predictor):
        """
        predictor : MacroAgent 전체 객체
        """
        self.predictor = predictor

        # Predictor 내부 함수 포워딩
        self.ticker = predictor.ticker
        self.window = predictor.window

        self.model = predictor.model
        self.scaler_X = predictor.scaler_X
        self.scaler_y = predictor.scaler_y
        self.feature_order = predictor.feature_order

        self.searcher = predictor.searcher
        self.predict = predictor.predict
        self.merge_data = predictor.merge_data
        self.feature_engineering = predictor.feature_engineering

        # DebateAgent 기본 상태
        self.agent_id = predictor.agent_id
        self.opinions = predictor.opinions if hasattr(predictor, "opinions") else []
        self.rebuttals = predictor.rebuttals if hasattr(predictor, "rebuttals") else {}


    # ==============================================================
    # 1) reviewer_draft — feature importance + LLM reasoning
    # ==============================================================
    def reviewer_draft(self, stock_data=None, target=None):

        # 1) StockData 준비
        if stock_data is None:
            stock_data = StockData(ticker=self.ticker)

        # 2) target 없으면 직접 예측
        if target is None:
            X_input = self.searcher(self.ticker)
            pred_dict = self.predict(X_input)

            target = Target(
                next_close=pred_dict["pred_return"],
                uncertainty=pred_dict["uncertainty"],
                confidence=pred_dict["confidence"],
            )

        # 3) 최신 feature 생성
        df = self.merge_data()
        df = self.feature_engineering(df)

        # 순서 맞추기
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_order]

        X_scaled = self.scaler_X.transform(df.values)
        X_scaled = X_scaled[-self.window :]

        # LSTM shape
        X_scaled = X_scaled.astype(np.float32).reshape(1, self.window, -1)

        # 4) GradientAnalyzer 실행
        feature_names = self.feature_order[:300]
        X_grad = X_scaled[:, :, :300]

        model = self.predictor.model.to("cpu")
        gradient_analyzer = GradientAnalyzer(model, feature_names)

        (
            importance_dict,
            temporal_df,
            consistency_df,
            sensitivity_df,
            grad_results,
        ) = gradient_analyzer.run_all_gradients(X_grad)

        # 요약 생성
        temporal_summary = (
            temporal_df.head().to_dict(orient="records") if temporal_df is not None else []
        )
        consistency_summary = (
            consistency_df.to_dict(orient="records") if consistency_df is not None else []
        )
        sensitivity_summary = (
            sensitivity_df.to_dict(orient="records") if sensitivity_df is not None else []
        )
        stability_summary = grad_results["stability_summary"]
        feature_summary = grad_results["feature_summary"]

        # 5) stock_data 저장 구조 구성
        setattr(
            stock_data,
            self.agent_id,
            {
                "feature_importance": {
                    "feature_summary": feature_summary,
                    "importance_dict": importance_dict,
                    "temporal_summary": temporal_summary,
                    "consistency_summary": consistency_summary,
                    "sensitivity_summary": sensitivity_summary,
                    "stability_summary": stability_summary,
                },
                "our_prediction": target.next_close,
                "uncertainty": round(target.uncertainty, 8),
                "confidence": round(target.confidence, 8),
            },
        )

        # 6) LLM Reason 생성
        sys_txt, usr_txt = self._build_messages_opinion(stock_data, target)

        parsed = self.predictor._ask_with_fallback(
            self.predictor._msg("system", sys_txt),
            self.predictor._msg("user", usr_txt),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        )

        reason = parsed.get("reason", "(사유 생성 실패)")

        op = Opinion(agent_id=self.agent_id, target=target, reason=reason)
        self.opinions.append(op)
        return op


    # ==============================================================
    # 2) reviewer_rebut — DebateAgent 대비 반박문 생성
    # ==============================================================
    def reviewer_rebut(self, my_opinion, other_opinion, round):

        sys_txt, usr_txt = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=my_opinion.stockdata if hasattr(my_opinion, "stockdata") else None,
        )

        parsed = self.predictor._ask_with_fallback(
            self.predictor._msg("system", sys_txt),
            self.predictor._msg("user", usr_txt),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"},
                },
                "required": ["stance", "message"],
            },
        )

        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 실패)"),
        )

        # 저장
        if round not in self.rebuttals:
            self.rebuttals[round] = []
        self.rebuttals[round].append(result)
        return result


    # ==============================================================
    # 3) reviewer_revise — β-weight 조정 + fine-tune + LLM revision
    # ==============================================================
    def reviewer_revise(self, my_opinion, others, rebuttals, stock_data,
                        fine_tune=True, lr=1e-4, epochs=5):

        gamma = getattr(self.predictor, "gamma", 0.3)

        # 1) β-weight 업데이트
        my_price = my_opinion.target.next_close
        my_sigma = abs(my_opinion.target.uncertainty)

        other_prices = np.array([o.target.next_close for o in others])
        other_sigmas = np.array([abs(o.target.uncertainty) for o in others])

        inv_sigmas = 1 / (np.concatenate([[my_sigma], other_sigmas]) + 1e-6)
        betas = inv_sigmas / inv_sigmas.sum()

        delta = np.sum(betas[1:] * (other_prices - my_price))
        revised_price = my_price + gamma * delta

        # 2) Fine-tuning (선택)
        loss_value = 0.0
        if fine_tune and isinstance(self.predictor.model, nn.Module):
            try:
                last_price = stock_data.last_price or my_price
                revised_ret = (revised_price / last_price) - 1.0
                revised_ret_scaled = self.scaler_y.transform([[revised_ret]])[0, 0]

                X_seq = self.searcher(self.ticker)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = self.predictor.model.to(device)
                model.train()

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

                X_t = torch.FloatTensor(X_seq).to(device)
                y_t = torch.FloatTensor([[revised_ret_scaled]]).to(device)

                for _ in range(epochs):
                    optimizer.zero_grad()
                    pred = model(X_t)
                    loss = criterion(pred, y_t)
                    loss.backward()
                    optimizer.step()
                    loss_value = loss.item()

                model.eval()
            except Exception as e:
                print(f"[ERROR] Fine-tune failed: {e}")

        # 3) Revision Reasoning
        new_target = Target(
            next_close=float(revised_price),
            uncertainty=my_opinion.target.uncertainty,
            confidence=my_opinion.target.confidence,
        )

        sys_txt, usr_txt = self._build_messages_revision(
            my_opinion=my_opinion,
            others=others,
            rebuttals=rebuttals,
            stock_data=stock_data,
        )

        parsed = self.predictor._ask_with_fallback(
            self.predictor._msg("system", sys_txt),
            self.predictor._msg("user", usr_txt),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        )

        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        op = Opinion(agent_id=self.agent_id, target=new_target, reason=revised_reason)
        self.opinions.append(op)

        print(f"[{self.agent_id}] revise → new_close={new_target.next_close:.4f}, loss={loss_value}")
        return op


    # ==============================================================
    # 내부: Opinion 프롬프트
    # ==============================================================
    def _build_messages_opinion(self, stock_data, target):

        agent_data = getattr(stock_data, self.agent_id)
        feature_imp = agent_data["feature_importance"]

        ctx = {
            "agent_id": self.agent_id,
            "ticker": stock_data.ticker,
            "last_price": stock_data.last_price,
            "our_prediction": target.next_close,
            "uncertainty": target.uncertainty,
            "confidence": target.confidence,
            "feature_importance": feature_imp,
        }

        sys = OPINION_PROMPTS[self.agent_id]["system"]
        usr = OPINION_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return sys, usr


    # ==============================================================
    # 내부: rebuttal 프롬프트
    # ==============================================================
    def _build_messages_rebuttal(self, my_opinion, target_opinion, stock_data):

        ctx = {
            "me": {
                "price": my_opinion.target.next_close,
                "reason": my_opinion.reason,
                "uncertainty": my_opinion.target.uncertainty,
                "confidence": my_opinion.target.confidence,
            },
            "other": {
                "price": target_opinion.target.next_close,
                "reason": target_opinion.reason,
                "uncertainty": target_opinion.target.uncertainty,
                "confidence": target_opinion.target.confidence,
            }
        }

        sys = REBUTTAL_PROMPTS[self.agent_id]["system"]
        usr = REBUTTAL_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return sys, usr


    # ==============================================================
    # 내부: revision 프롬프트
    # ==============================================================
    def _build_messages_revision(self, my_opinion, others, rebuttals, stock_data):

        others_summary = []
        for o in others:
            entry = {
                "agent_id": o.agent_id,
                "price": o.target.next_close,
                "reason": o.reason,
                "uncertainty": o.target.uncertainty,
                "confidence": o.target.confidence,
            }
            others_summary.append(entry)

        ctx = {
            "my_opinion": {
                "price": my_opinion.target.next_close,
                "reason": my_opinion.reason,
                "uncertainty": my_opinion.target.uncertainty,
                "confidence": my_opinion.target.confidence,
            },
            "others": others_summary,
        }

        sys = REVISION_PROMPTS[self.agent_id]["system"]
        usr = REVISION_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return sys, usr
