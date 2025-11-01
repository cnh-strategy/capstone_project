# core\sentimental_classes\builders.py
from typing import Dict, Any

def _round(x, n=4):
    try: return None if x is None else round(float(x), n)
    except: return None

def build_prediction_block(stock_data, target):
    last = float(getattr(stock_data, "last_price", 0.0) or 0.0)
    pred_close = float(getattr(target, "next_close", 0.0) or 0.0)
    pred_return = getattr(target, "pred_return", None)
    if pred_return is None and last > 0:
        pred_return = (pred_close - last) / last
    unc = getattr(target, "uncertainty", 0.0) or 0.0
    std = _round(unc.get("std")) if isinstance(unc, dict) else _round(unc)
    ci95 = unc.get("ci95") if isinstance(unc, dict) else None
    pi80 = unc.get("pi80") if isinstance(unc, dict) else None
    conf = _round(getattr(target, "confidence", 0.0), 3)
    prob_up = getattr(target, "calibrated_prob_up", None)
    return {
        "pred_close": _round(pred_close),
        "pred_return": _round(pred_return),
        "uncertainty": {"std": std, "ci95": ci95, "pi80": pi80},
        "confidence": conf,
        "calibrated_prob_up": _round(prob_up, 3) if prob_up is not None else None,
        "mc_mean_next_close": _round(getattr(target, "mc_mean_next_close", None)),
        "mc_std": _round(getattr(target, "mc_std", None)),
    }
