# app/services/logger.py

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "query_log.jsonl"


def log_query(
    question: str,
    answer: str,
    verses: List[Dict[str, Any]],
) -> None:
    """Append one query-answer record to a JSONL log file."""
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "answer": answer,
        "verses": verses,  # list of {book, chapter, verse, text, ...}
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")