# debate_ver3/agents/test_import.py
# ===============================================================

import importlib
m = importlib.import_module('debate_ver3.agents.sentimental_agent')
print("loaded:", m is not None, "| has class:", hasattr(m, 'SentimentalAgent'))
