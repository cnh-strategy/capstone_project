from datetime import datetime
from typing import Optional, Union

def safe_parse_iso_datetime(s: Optional[Union[str, float, int]]) -> Optional[datetime]:
    # None/숫자/빈문자열 방어
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # 'Z' → '+00:00' 보정
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None
