#!/usr/bin/env python3
"""
신뢰도 계산 상세 테스트 - MacroAgent의 confidence가 1.0인 이유 확인
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agents.macro_agent import MacroAgent
from config.agents import dir_info, common_params
import pandas as pd
import numpy as np
import torch

def test_confidence_detail(ticker="AAPL"):
    """MacroAgent의 confidence 계산 상세 확인"""
    print("=" * 80)
    print(f"MacroAgent 신뢰도 계산 상세 테스트 - Ticker: {ticker}")
    print("=" * 80)
    print()
    
    agent = MacroAgent(
        agent_id="MacroAgent",
        ticker=ticker,
        data_dir=dir_info["data_dir"],
        model_dir=dir_info["model_dir"]
    )
    
    # 데이터 검색
    agent.search(ticker=ticker)
    
    # 모델 로드
    model_path = os.path.join(dir_info["model_dir"], f"{ticker}_MacroAgent.pt")
    if os.path.exists(model_path):
        agent.load_model()
    
    # 데이터셋 로드
    dataset_path = os.path.join(dir_info["data_dir"], f"{ticker}_MacroAgent_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"❌ 데이터셋이 없습니다: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    lookback_days = common_params.get("confidence_lookback_days", 30)
    
    meta_cols = {"sample_id", "time_step", "target", "date"}
    feature_cols = [
        c for c in df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    unique_samples = sorted(df['sample_id'].unique())
    if len(unique_samples) < lookback_days:
        print(f"❌ 샘플이 부족합니다: {len(unique_samples)} < {lookback_days}")
        return
    
    recent_samples = unique_samples[-lookback_days:]
    
    print(f"분석 대상:")
    print(f"  - 전체 샘플 수: {len(unique_samples)}")
    print(f"  - 최근 {lookback_days}개 샘플: {len(recent_samples)}개")
    print(f"  - 샘플 ID 범위: {recent_samples[0]} ~ {recent_samples[-1]}")
    print()
    
    # 모델 확인
    if agent.model is None:
        if hasattr(agent, 'forward') and isinstance(agent, torch.nn.Module):
            model_to_use = agent
        else:
            print("❌ 모델을 사용할 수 없습니다")
            return
    else:
        model_to_use = agent.model
    
    print(f"모델 정보:")
    print(f"  - model_to_use: {type(model_to_use)}")
    print(f"  - agent.model: {agent.model}")
    print(f"  - y_scaler 존재: {hasattr(agent.scaler, 'y_scaler') and agent.scaler.y_scaler is not None}")
    print()
    
    # 디바이스 설정
    if hasattr(agent, "device"):
        device = agent.device
    elif hasattr(model_to_use, "parameters"):
        try:
            device = next(model_to_use.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    print(f"디바이스: {device}")
    print()
    
    model_to_use.eval()
    
    correct_count = 0
    total_count = 0
    details = []
    
    print("각 샘플별 방향 정확도 확인:")
    print("-" * 80)
    
    for sample_id in recent_samples:
        try:
            sample_data = df[df['sample_id'] == sample_id].sort_values('time_step')
            if len(sample_data) == 0:
                continue
            
            X_sample = sample_data[feature_cols].values.astype(np.float32)
            y_actual = sample_data['target'].iloc[-1]
            
            if np.isnan(y_actual) or np.any(np.isnan(X_sample)):
                continue
            
            X_tensor = torch.from_numpy(X_sample).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model_to_use(X_tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                y_pred_raw = out.detach().cpu().numpy().squeeze()
            
            # y_actual 역변환 (스케일링된 값을 원본 수익률로)
            y_actual_original = y_actual
            if hasattr(agent, "scaler") and hasattr(agent.scaler, "y_scaler") and agent.scaler.y_scaler is not None:
                try:
                    y_actual_scaled = np.array([[y_actual]])
                    y_actual_inverse = agent.scaler.inverse_y(y_actual_scaled)[0, 0]
                    # 역변환 결과 확인
                    if abs(y_actual_inverse) < 1.0:  # 원본 수익률은 보통 -1~1 범위
                        y_actual = y_actual_inverse
                except Exception as e:
                    # 역변환 실패 시 원본 사용
                    pass
            
            # y_scaler 역변환
            y_pred = y_pred_raw
            if hasattr(agent, "scaler") and hasattr(agent.scaler, "y_scaler") and agent.scaler.y_scaler is not None:
                try:
                    y_pred_scaled = np.array([[y_pred_raw]])
                    y_pred = agent.scaler.inverse_y(y_pred_scaled)[0, 0]
                except Exception:
                    pass
            
            pred_sign = np.sign(y_pred)
            actual_sign = np.sign(y_actual)
            is_correct = (pred_sign == actual_sign)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            details.append({
                'sample_id': sample_id,
                'y_actual_original': y_actual_original,
                'y_actual': y_actual,
                'y_pred_raw': y_pred_raw,
                'y_pred': y_pred,
                'pred_sign': pred_sign,
                'actual_sign': actual_sign,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"  샘플 {sample_id} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n결과 요약:")
    print(f"  - 총 샘플 수: {total_count}")
    print(f"  - 정확한 예측: {correct_count}")
    print(f"  - 정확도: {correct_count / total_count if total_count > 0 else 0:.4f}")
    print()
    
    # target 값 분포 확인
    actual_original_values = [d['y_actual_original'] for d in details]
    actual_values = [d['y_actual'] for d in details]
    pred_values = [d['y_pred'] for d in details]
    
    print(f"Target 값 분포:")
    print(f"  - y_actual (스케일링된 원본) 범위: [{min(actual_original_values):.6f}, {max(actual_original_values):.6f}]")
    print(f"  - y_actual (역변환 후) 범위: [{min(actual_values):.6f}, {max(actual_values):.6f}]")
    print(f"  - y_actual (역변환 후) 평균: {np.mean(actual_values):.6f}")
    print(f"  - y_actual (역변환 후) 표준편차: {np.std(actual_values):.6f}")
    print(f"  - y_actual (역변환 후) 부호 분포: 양수={sum(1 for v in actual_values if v > 0)}, 음수={sum(1 for v in actual_values if v < 0)}, 0={sum(1 for v in actual_values if v == 0)}")
    print()
    print(f"  - y_pred 범위: [{min(pred_values):.6f}, {max(pred_values):.6f}]")
    print(f"  - y_pred 평균: {np.mean(pred_values):.6f}")
    print(f"  - y_pred 표준편차: {np.std(pred_values):.6f}")
    print(f"  - y_pred 부호 분포: 양수={sum(1 for v in pred_values if v > 0)}, 음수={sum(1 for v in pred_values if v < 0)}, 0={sum(1 for v in pred_values if v == 0)}")
    print()
    
    # 처음 10개와 마지막 10개 상세 출력
    print("처음 10개 샘플:")
    for i, detail in enumerate(details[:10]):
        status = "✅" if detail['correct'] else "❌"
        print(f"  {status} 샘플 {detail['sample_id']}: "
              f"예측={detail['y_pred']:.6f} ({detail['pred_sign']:+.0f}), "
              f"실제={detail['y_actual']:.6f} ({detail['actual_sign']:+.0f})")
    
    if len(details) > 20:
        print("  ...")
        print("마지막 10개 샘플:")
        for detail in details[-10:]:
            status = "✅" if detail['correct'] else "❌"
            print(f"  {status} 샘플 {detail['sample_id']}: "
                  f"예측={detail['y_pred']:.6f} ({detail['pred_sign']:+.0f}), "
                  f"실제={detail['y_actual']:.6f} ({detail['actual_sign']:+.0f})")
    
    # 문제가 있는 샘플 확인
    incorrect_samples = [d for d in details if not d['correct']]
    if incorrect_samples:
        print(f"\n❌ 부정확한 샘플 ({len(incorrect_samples)}개):")
        for detail in incorrect_samples[:5]:
            print(f"  샘플 {detail['sample_id']}: "
                  f"예측={detail['y_pred']:.6f} ({detail['pred_sign']:+.0f}), "
                  f"실제={detail['y_actual']:.6f} ({detail['actual_sign']:+.0f})")
    
    print()
    print("=" * 80)
    print(f"최종 confidence: {correct_count / total_count if total_count > 0 else 0:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="신뢰도 계산 상세 테스트")
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="테스트할 티커 (기본값: AAPL)"
    )
    
    args = parser.parse_args()
    test_confidence_detail(args.ticker)

