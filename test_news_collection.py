#!/usr/bin/env python3
"""
뉴스 데이터 수집 기능 테스트 스크립트
"""
import os
import sys
from pathlib import Path
from datetime import date, timedelta

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("뉴스 데이터 수집 기능 테스트")
print("=" * 80)

# 1. 환경변수 확인
print("\n1️⃣ 환경변수 확인")
eodhd_key = os.getenv("EODHD_API_KEY")
if eodhd_key:
    print(f"✅ EODHD_API_KEY 설정됨: {eodhd_key[:10]}...")
else:
    print("⚠️  EODHD_API_KEY가 설정되지 않았습니다.")
    print("   .env 파일에 EODHD_API_KEY를 설정하거나 환경변수로 설정하세요.")

# 2. 모듈 import 테스트
print("\n2️⃣ 모듈 import 테스트")
try:
    from core.sentimental_classes.finbert_utils import (
        load_or_fetch_news,
        get_news_cache_path,
        _normalize_symbol,
    )
    from core.sentimental_classes.eodhd_client import EODHDNewsClient
    print("✅ 모듈 import 성공")
except Exception as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)

# 3. EODHD 클라이언트 테스트
print("\n3️⃣ EODHD 클라이언트 초기화 테스트")
try:
    if eodhd_key:
        client = EODHDNewsClient(api_key=eodhd_key)
        print("✅ EODHDNewsClient 초기화 성공")
    else:
        print("⚠️  API 키가 없어 클라이언트 초기화를 건너뜁니다.")
        client = None
except Exception as e:
    print(f"❌ EODHDNewsClient 초기화 실패: {e}")
    client = None

# 4. 뉴스 수집 테스트
print("\n4️⃣ 뉴스 수집 테스트")
ticker = "NVDA"
end_date = date.today() - timedelta(days=1)  # 어제
start_date = end_date - timedelta(days=40)  # 40일 전

print(f"   티커: {ticker}")
print(f"   기간: {start_date} ~ {end_date}")

try:
    # 캐시 경로 확인
    cache_path = get_news_cache_path(ticker, start_date, end_date)
    print(f"   캐시 경로: {cache_path}")
    print(f"   캐시 존재 여부: {cache_path.exists()}")
    
    if cache_path.exists():
        print(f"   ✅ 캐시 파일이 이미 존재합니다: {cache_path}")
        # 파일 크기 확인
        size = cache_path.stat().st_size
        print(f"   파일 크기: {size:,} bytes")
        
        # 파일 내용 일부 확인
        import json
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"   뉴스 개수: {len(data)}건")
            if len(data) > 0:
                print(f"   첫 번째 뉴스 샘플:")
                first = data[0]
                print(f"     - 날짜: {first.get('date', 'N/A')}")
                print(f"     - 제목: {first.get('title', 'N/A')[:50]}...")
        else:
            print(f"   ⚠️  캐시 파일 형식이 올바르지 않습니다 (list가 아님)")
    else:
        print(f"   ⚠️  캐시 파일이 없습니다.")
        
        # 실제 수집 시도
        if eodhd_key:
            print(f"\n   📥 뉴스 수집 시도 중...")
            news_data = load_or_fetch_news(
                ticker=ticker,
                start=start_date,
                end=end_date,
                api_key=eodhd_key
            )
            
            if news_data and len(news_data) > 0:
                print(f"   ✅ 뉴스 수집 성공: {len(news_data)}건")
                print(f"   ✅ 캐시 파일 생성됨: {cache_path}")
                
                # 첫 번째 뉴스 샘플 출력
                if len(news_data) > 0:
                    first = news_data[0]
                    print(f"\n   첫 번째 뉴스 샘플:")
                    print(f"     - 날짜: {first.get('date', 'N/A')}")
                    print(f"     - 제목: {first.get('title', 'N/A')[:80]}")
                    print(f"     - 출처: {first.get('source', 'N/A')}")
            else:
                print(f"   ⚠️  뉴스 수집 결과: 0건 (또는 수집 실패)")
        else:
            print(f"   ⚠️  API 키가 없어 뉴스 수집을 건너뜁니다.")
            
except Exception as e:
    print(f"   ❌ 뉴스 수집 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 5. build_finbert_news_features 테스트
print("\n5️⃣ build_finbert_news_features 테스트")
try:
    from agents.sentimental_agent import build_finbert_news_features
    
    asof_date = date.today().isoformat()
    print(f"   기준 날짜: {asof_date}")
    
    feats = build_finbert_news_features(
        ticker=ticker,
        asof_kst=asof_date,
        base_dir=os.path.join("data", "raw", "news")
    )
    
    print(f"   ✅ 피처 생성 완료")
    print(f"\n   생성된 피처:")
    print(f"     - has_news: {feats.get('has_news', False)}")
    print(f"     - 7일 평균 감성: {feats.get('sentiment_summary', {}).get('mean_7d', 0.0):.4f}")
    print(f"     - 7일 뉴스 개수: {feats.get('news_count', {}).get('count_7d', 0)}")
    print(f"     - 7일 감성 변동성: {feats.get('sentiment_volatility', {}).get('vol_7d', 0.0):.4f}")
    print(f"     - 7일 추세: {feats.get('trend_7d', 0.0):.4f}")
    
except Exception as e:
    print(f"   ❌ build_finbert_news_features 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 6. 캐시 디렉토리 확인
print("\n6️⃣ 캐시 디렉토리 확인")
cache_dir = project_root / "data" / "raw" / "news"
print(f"   캐시 디렉토리: {cache_dir}")
print(f"   디렉토리 존재: {cache_dir.exists()}")

if cache_dir.exists():
    json_files = list(cache_dir.glob("*.json"))
    print(f"   JSON 파일 개수: {len(json_files)}")
    if len(json_files) > 0:
        print(f"   파일 목록:")
        for f in sorted(json_files)[:5]:  # 최대 5개만
            size = f.stat().st_size
            print(f"     - {f.name} ({size:,} bytes)")
        if len(json_files) > 5:
            print(f"     ... 외 {len(json_files) - 5}개 파일")
else:
    print(f"   ⚠️  캐시 디렉토리가 없습니다.")

print("\n" + "=" * 80)
print("테스트 완료")
print("=" * 80)

