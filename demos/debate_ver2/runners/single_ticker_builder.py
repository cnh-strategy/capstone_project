#!/usr/bin/env python3
"""
Single Ticker Dataset Builder
개별 티커 데이터를 yfinance에서 수집하고 전처리 후 저장
(Pretraining 전에 반드시 실행)
"""
import os
from datetime import datetime
from core.preprocessing import build_dataset

# ---------------------------------------------------------
# 1️⃣ 단일 티커 데이터셋 생성
# ---------------------------------------------------------
def build_single_ticker(ticker="TSLA", save_dir="data/processed", window=7):
    print("\n📈 [Single Ticker Dataset Builder]")
    print("=" * 60)
    X, y, sx, sy = build_dataset(ticker, save_dir=save_dir, window_size=window)

    # 저장 경로 확인
    npz_path = os.path.join(save_dir, f"{ticker}_dataset.npz")
    if os.path.exists(npz_path):
        print(f"✅ Dataset created and saved: {npz_path}")
        print(f"Samples: {len(X)} | Window: {window} | Features: {X.shape[-1]}")
    else:
        print("❌ Dataset save failed — check file permissions.")
    print("=" * 60)
    print("✅ Dataset Build Completed.\n")

# ---------------------------------------------------------
# 2️⃣ CLI Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset for a single stock ticker")
    parser.add_argument("--ticker", type=str, default="TSLA", help="데이터셋 생성할 티커 코드")
    parser.add_argument("--window", type=int, default=7, help="시퀀스 윈도우 크기")
    args = parser.parse_args()

    build_single_ticker(args.ticker, window=args.window)
