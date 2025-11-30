import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from agents.sentimental_agent import SentimentalAgent

# 백테스트용 경로 설정
BACKTEST_DATA_DIR = os.path.join(project_root, "backtest", "data", "processed")
os.makedirs(BACKTEST_DATA_DIR, exist_ok=True)

def prefetch_news():
    tickers = ["NVDL", "AAPL", "TSLA"]
    print(f"🚀 뉴스 데이터 사전 구축 시작: {tickers}")
    print("   (이 작업은 GPU를 사용하여 감성 분석을 수행하므로 시간이 걸립니다.)")

    for ticker in tickers:
        print(f"\n[{ticker}] 뉴스 수집 및 분석 중...")
        try:
            # SentimentalAgent를 생성하면 내부적으로 DB 경로를 설정함
            agent = SentimentalAgent(
                ticker=ticker, 
                data_dir=BACKTEST_DATA_DIR
            )
            # _ensure_sentimental_csv -> update_news_db 호출
            # rebuild=False로 설정하여 이미 존재하는 경우 건너뛰도록 함 (중복 실행 방지)
            # 하지만 최초 실행이거나 확실한 업데이트를 원하면 rebuild=True 권장
            # 여기서는 DB가 삭제되었으므로 rebuild=True와 동일한 효과
            agent._ensure_sentimental_csv(ticker, rebuild=True)
            print(f"✅ {ticker} 완료")
        except Exception as e:
            print(f"❌ {ticker} 실패: {e}")

if __name__ == "__main__":
    prefetch_news()

