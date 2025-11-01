
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_opinion_with_without_ctx.py (improved)

Adds robust fallbacks:
- If agent.opinion() has no use_ctx flag, we monkey-patch agent.build_message_opinion()
  to strip the Context block for "pre-ctx" and restore for "with-ctx".
- Tries multiple signatures for build_message_opinion:
    (), (use_ctx=True/False), (ticker=...), (use_ctx=..., ticker=...)
- If build_message_opinion returns a dict or object, extract system/user strings smartly.
- LLM call fallbacks extended.
"""

import argparse
import importlib
import inspect
import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Tuple

def _to_dict(x):
    if x is None:
        return None
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: _to_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_dict(v) for v in x]
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)

def _strip_ctx_text(user_prompt: str) -> str:
    try:
        without = re.sub(r"Context:\s*\{.*?\}\s*", "", user_prompt, flags=re.DOTALL)
    except re.error:
        without = user_prompt
    without = without.strip() or "아래 정보를 바탕으로 간결히 의견을 3문장 이내로 작성해줘."
    return without

def _extract_msgs(ret) -> Tuple[str, str]:
    """
    Accepts tuple, dict, or object. Returns (system, user) strings.
    """
    if isinstance(ret, tuple) and len(ret) >= 2:
        return str(ret[0]), str(ret[1])
    if isinstance(ret, dict):
        sysmsg = ret.get("system") or ret.get("system_msg") or ret.get("system_message") or ""
        usrmsg = ret.get("user") or ret.get("user_msg") or ret.get("user_message") or ret.get("prompt") or ""
        return str(sysmsg), str(usrmsg)
    # object-like with attributes
    for s_key, u_key in [("system","user"), ("system_msg","user_msg"), ("system_message","user_message")]:
        if hasattr(ret, s_key) and hasattr(ret, u_key):
            return str(getattr(ret, s_key)), str(getattr(ret, u_key))
    raise RuntimeError("build_message_opinion did not return recognizable messages. Expected (system, user) tuple or dict with keys.")

def _try_build_message_opinion(agent, *, want_ctx: bool):
    """
    Try calling build_message_opinion with various signatures.
    Returns (system, user).
    """
    if not hasattr(agent, "build_message_opinion") or not callable(agent.build_message_opinion):
        raise RuntimeError("No build_message_opinion method found.")

    candidates = []
    # 1) no args
    candidates.append(({}, {}))
    # 2) use_ctx flag only
    candidates.append(({}, {"use_ctx": want_ctx}))
    # 3) ticker only
    if getattr(agent, "ticker", None) is not None:
        candidates.append(({}, {"ticker": agent.ticker}))
        candidates.append(({}, {"use_ctx": want_ctx, "ticker": agent.ticker}))

    last_err = None
    for args, kwargs in candidates:
        try:
            ret = agent.build_message_opinion(*args, **kwargs)  # type: ignore
            return _extract_msgs(ret)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"build_message_opinion call failed with all tried signatures. Last error: {last_err}")

def _call_llm_like(agent, system_msg, user_msg):
    for fn in ["call_llm", "_call_llm", "infer_opinion", "generate_text", "llm_call"]:
        if hasattr(agent, fn) and callable(getattr(agent, fn)):
            return getattr(agent, fn)(system_msg, user_msg)
    # underlying llm
    if hasattr(agent, "llm"):
        llm = getattr(agent, "llm")
        for fn in ["generate", "chat", "call", "complete"]:
            if hasattr(llm, fn) and callable(getattr(llm, fn)):
                return getattr(llm, fn)(system_msg, user_msg)
    # __call__
    if hasattr(agent, "__call__"):
        try:
            return agent(system_msg, user_msg)  # type: ignore
        except Exception:
            pass
    raise RuntimeError("No way to call the LLM. Expose call_llm(system, user) or similar.")

def _format_opinion_result(raw) -> Dict[str, Any]:
    if raw is None:
        return {"text": None}
    if is_dataclass(raw):
        raw = asdict(raw)
    if isinstance(raw, dict):
        out = {}
        out["agent_id"] = raw.get("agent_id") or raw.get("agent") or raw.get("name")
        out["text"] = raw.get("reason") or raw.get("text") or raw.get("opinion") or raw.get("content")
        out["target"] = raw.get("target")
        out["prediction"] = raw.get("prediction")
        out["confidence"] = raw.get("confidence")
        out["uncertainty"] = raw.get("uncertainty")
        out["pre_post"] = raw.get("pre_post")
        for k, v in raw.items():
            if k not in out:
                out[k] = v
        return _to_dict(out)
    return {"text": str(raw)}

def _pretty_json(d):
    return json.dumps(d, ensure_ascii=False, indent=2)

def _side_by_side_diff(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    keys = sorted(set(list(a.keys()) + list(b.keys())))
    lines = []
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if av != bv:
            lines.append(f"- {k}:\n    pre-ctx : {av}\n    with-ctx: {bv}")
    if not lines:
        return "(No differences found — the two opinions are identical by these keys.)"
    return "\n".join(lines)

def opinion_pre_ctx(agent) -> Dict[str, Any]:
    # Path A: dedicated flag
    for fn in ["opinion", "generate_opinion"]:
        if hasattr(agent, fn) and callable(getattr(agent, fn)):
            sig = inspect.signature(getattr(agent, fn))
            if "use_ctx" in sig.parameters:
                raw = getattr(agent, fn)(use_ctx=False)
                return _format_opinion_result(raw)

    # Path B: monkey-patch build_message_opinion used by .opinion()
    if hasattr(agent, "opinion") and callable(agent.opinion) and hasattr(agent, "build_message_opinion"):
        orig = agent.build_message_opinion
        def wrapper(*args, **kwargs):
            ret = orig(*args, **kwargs)
            sysmsg, usrmsg = _extract_msgs(ret)
            usrmsg = _strip_ctx_text(usrmsg)
            return sysmsg, usrmsg
        agent.build_message_opinion = wrapper  # type: ignore
        try:
            raw = agent.opinion()
            return _format_opinion_result(raw)
        finally:
            agent.build_message_opinion = orig  # restore

    # Path C: message-level call
    if hasattr(agent, "build_message_opinion"):
        sysmsg, usrmsg = _try_build_message_opinion(agent, want_ctx=False)
        usrmsg = _strip_ctx_text(usrmsg)
        raw = _call_llm_like(agent, sysmsg, usrmsg)
        return _format_opinion_result(raw)

    raise RuntimeError("Could not compute pre-ctx opinion. Expose .opinion(use_ctx=False) or .build_message_opinion().")

def opinion_with_ctx(agent) -> Dict[str, Any]:
    # Path A: dedicated flag
    for fn in ["opinion", "generate_opinion"]:
        if hasattr(agent, fn) and callable(getattr(agent, fn)):
            sig = inspect.signature(getattr(agent, fn))
            if "use_ctx" in sig.parameters:
                raw = getattr(agent, fn)(use_ctx=True)
                return _format_opinion_result(raw)

    # Path B: normal opinion() (assuming it uses ctx internally)
    if hasattr(agent, "opinion") and callable(agent.opinion):
        try:
            raw = agent.opinion()
            return _format_opinion_result(raw)
        except Exception:
            pass

    # Path C: message-level call
    if hasattr(agent, "build_message_opinion"):
        sysmsg, usrmsg = _try_build_message_opinion(agent, want_ctx=True)
        raw = _call_llm_like(agent, sysmsg, usrmsg)
        return _format_opinion_result(raw)

    raise RuntimeError("Could not compute with-ctx opinion. Expose .opinion(use_ctx=True) or .build_message_opinion().")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    p.add_argument("--class", dest="cls", required=True)
    p.add_argument("--ticker", required=True)
    p.add_argument("--kwargs", default="{}")
    args = p.parse_args()

    mod = importlib.import_module(args.module)
    AgentCls = getattr(mod, args.cls)
    extra_kwargs = json.loads(args.kwargs)

    # Instantiate
    try:
        agent = AgentCls(ticker=args.ticker, **extra_kwargs)
    except TypeError:
        agent = AgentCls(**extra_kwargs)
        if getattr(agent, "ticker", None) is None:
            if hasattr(agent, "set_ticker") and callable(agent.set_ticker):
                agent.set_ticker(args.ticker)
            else:
                setattr(agent, "ticker", args.ticker)

    pre_ctx = opinion_pre_ctx(agent)
    with_ctx = opinion_with_ctx(agent)

    print("\n=== PRE-CTX (ctx 제거) ===")
    print(_pretty_json(pre_ctx))

    print("\n=== WITH-CTX (현재 ctx 적용) ===")
    print(_pretty_json(with_ctx))

    print("\n=== FIELD DIFFS ===")
    print(_side_by_side_diff(pre_ctx, with_ctx))

    pre_txt = (pre_ctx.get("text") or "").strip().replace("\n", " ")
    with_txt = (with_ctx.get("text") or "").strip().replace("\n", " ")
    print("\n=== SUMMARY ===")
    print(f"pre-ctx : {pre_txt}")
    print(f"with-ctx: {with_txt}")

if __name__ == "__main__":
    main()
