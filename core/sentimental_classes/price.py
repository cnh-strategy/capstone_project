# core\sentimental_classes\price.py

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from debate_ver3.agents.base_agent import StockData

def _ensure_ctx_features_from_window(self, stock_data: StockData):
    X_last = getattr(self, "_last_X", None)
    fcols = getattr(stock_data, "feature_cols", None)
    if X_last is None or fcols is None or len(fcols) == 0:
        return

    try:
        import numpy as np
        XF = X_last[0] if X_last.ndim == 3 else X_last

        def _ix(name: str) -> int:
            try:
                return fcols.index(name)
            except ValueError:
                return -1

        ix_close = _ix("Close")
        ix_open = _ix("Open")
        ix_high = _ix("High")
        ix_low = _ix("Low")
        ix_vol = _ix("Volume")
        ix_ret = _ix("returns")

        close = XF[:, ix_close] if ix_close >= 0 else None
        high = XF[:, ix_high] if ix_high >= 0 else None
        low = XF[:, ix_low] if ix_low >= 0 else None
        prevc = np.roll(close, 1) if close is not None else None
        vol = XF[:, ix_vol] if ix_vol >= 0 else None
        ret = XF[:, ix_ret] if ix_ret >= 0 else None

        def pct_change(a, lag=1):
            if a is None:
                return None
            b = np.array(a, dtype=float)
            c = np.full_like(b, np.nan)
            c[lag:] = (b[lag:] - b[:-lag]) / np.where(b[:-lag] == 0, np.nan, b[:-lag])
            return c

        def rolling_mean(a, n):
            if a is None:
                return None
            import numpy as _np
            b = _np.array(a, dtype=float)
            out = _np.full_like(b, _np.nan)
            for i in range(n - 1, len(b)):
                out[i] = _np.nanmean(b[i - n + 1:i + 1])
            return out

        def rolling_std(a, n):
            if a is None:
                return None
            import numpy as _np
            b = _np.array(a, dtype=float)
            out = _np.full_like(b, _np.nan)
            for i in range(n - 1, len(b)):
                out[i] = _np.nanstd(b[i - n + 1:i + 1], ddof=0)
            return out

        if close is not None:
            if ret is None:
                ret = pct_change(close, 1)

            lag_ret_1 = np.nan if ret is None else ret[-2]
            lag_ret_5 = np.nan if ret is None else np.nanmean(ret[-6:-1]) if len(ret) >= 6 else np.nan
            lag_ret_20 = np.nan if ret is None else np.nanmean(ret[-21:-1]) if len(ret) >= 21 else np.nan

            trend_7d = pct_change(close, 7)
            rolling_vol_20 = rolling_std(ret, 20) if ret is not None else None

            if (high is not None) and (low is not None):
                hl = np.abs(high - low)
                hc = np.abs(high - prevc) if prevc is not None else None
                lc = np.abs(low - prevc) if prevc is not None else None
                if hc is not None and lc is not None:
                    import numpy as _np
                    tr = _np.nanmax(_np.vstack([hl, hc, lc]), axis=0)
                    atr_14_series = rolling_mean(tr, 14)
                    atr_14_val = atr_14_series[-1] if atr_14_series is not None else np.nan
                else:
                    atr_14_val = np.nan
            else:
                atr_14_val = np.nan

            mean20 = rolling_mean(close, 20)
            std20 = rolling_std(close, 20)
            z20 = (close - mean20) / std20 if (mean20 is not None and std20 is not None) else None

            if close is not None:
                import numpy as _np
                dd = _np.full_like(close, _np.nan)
                mx = _np.full_like(close, _np.nan)
                for i in range(len(close)):
                    start = max(0, i - 19)
                    mx[i] = _np.nanmax(close[start:i + 1])
                    dd[i] = close[i] / mx[i] - 1.0 if mx[i] not in (0, np.nan) else _np.nan
                drawdown_20_series = dd
                breakout_20_flag = bool(close[-1] > mx[-2]) if len(close) >= 21 and not np.isnan(mx[-2]) else False
            else:
                z20 = None
                drawdown_20_series = None
                breakout_20_flag = False

            if vol is not None:
                vm20 = rolling_mean(vol, 20)
                vs20 = rolling_std(vol, 20)
                vol_z20_series = (vol - vm20) / vs20 if (vm20 is not None and vs20 is not None) else None
                vol_z20_val = vol_z20_series[-1] if vol_z20_series is not None else np.nan
                denom = vm20[-1] if vm20 is not None else np.nan
                turnover_rate_val = float(vol[-1] / denom) if denom and not np.isnan(denom) and denom != 0 else 0.0
                volume_spike_flag = bool(vol_z20_val >= 2.0) if not np.isnan(vol_z20_val) else False
            else:
                vol_z20_val = np.nan
                turnover_rate_val = 0.0
                volume_spike_flag = False

            stock_data.__dict__.update({
                "lag_ret_1": None if np.isnan(lag_ret_1) else float(lag_ret_1),
                "lag_ret_5": None if np.isnan(lag_ret_5) else float(lag_ret_5),
                "lag_ret_20": None if np.isnan(lag_ret_20) else float(lag_ret_20),
                "rolling_vol_20": None if (rolling_vol_20 is None or np.isnan(rolling_vol_20[-1])) else float(rolling_vol_20[-1]),
                "trend_7d": None if (trend_7d is None or np.isnan(trend_7d[-1])) else float(trend_7d[-1]),
                "atr_14": None if np.isnan(atr_14_val) else float(atr_14_val),
                "breakout_20": bool(breakout_20_flag),
                "zscore_close_20": None if (z20 is None or np.isnan(z20[-1])) else float(z20[-1]),
                "drawdown_20": None if (drawdown_20_series is None or np.isnan(drawdown_20_series[-1])) else float(drawdown_20_series[-1]),
                "vol_zscore_20": None if np.isnan(vol_z20_val) else float(vol_z20_val),
                "turnover_rate": float(turnover_rate_val),
                "volume_spike": bool(volume_spike_flag),
            })
    except Exception:
        pass
