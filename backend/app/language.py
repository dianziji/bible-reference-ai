"""Which language a question is in, and therefore which translation answers it.

Only English and Chinese are supported, so detection is a CJK check: any
Han character makes the question Chinese. Deterministic, dependency-free,
and wrong only for the edge case of an English question quoting a Chinese
word — which then gets a perfectly readable 和合本 answer.
"""

import re

_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

LANGUAGE_NAMES = {"en": "English", "zh": "Chinese (简体中文)"}


def detect_language(text: str) -> str:
    return "zh" if _HAN.search(text) else "en"
