import os
import json
import requests
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# API KEY 로드
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
if not EODHD_API_KEY:
    raise RuntimeError(
        "EODHD_API_KEY is missing. Add it to your .env file: EODHD_API_KEY=xxx"
    )

def fetch_history_news(ticker, start, end):
    url = (
        f"https://eodhd.com/api/news?"
        f"s={ticker}&from={start}&to={end}&api_token={EODHD_API_KEY}&fmt=json"
    )

    r = requests.get(url, timeout=10)

    # 요청 실패
    if not r.ok:
        raise RuntimeError(f"[fetch_history_news] HTTP {r.status_code}: {r.text}")

    # JSON parse
    try:
        data = r.json()
    except json.JSONDecodeError:
        raise RuntimeError(
            f"[fetch_history_news] Invalid JSON response:\n{r.text[:300]}"
        )

    # API 오류 메시지
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"[fetch_history_news] API Error: {data}")

    return data
