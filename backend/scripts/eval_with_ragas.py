# scripts/eval_with_ragas.py

import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    # context_precision,  # 需要 reference 列（ground truth）
    # context_recall,     # 需要 reference 列（ground truth）
)

# 加载 .env 文件中的环境变量（包括 OPENAI_API_KEY）
load_dotenv()

# 这里直接用相对路径指向 app/logs/query_log.jsonl
# eval_with_ragas.py -> scripts -> backend(root) -> app -> logs -> query_log.jsonl
LOG_PATH = (
    Path(__file__)
    .resolve()
    .parent   # scripts
    .parent   # backend 根目录
    / "app"
    / "logs"
    / "query_log.jsonl"
)


def load_logs(max_samples: int | None = None):
    """从日志文件中读取若干条记录，用于评估。"""
    if not LOG_PATH.exists():
        print(f"⚠️  Log file not found: {LOG_PATH}")
        print("   请先通过前端或 curl 调用 /api/query 生成一些日志，再运行评估脚本。")
        return []

    records = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

    if max_samples is not None:
        records = records[:max_samples]

    return records


def build_ragas_dataset(records):
    """
    Ragas 需要的字段通常是：
    - question
    - answer
    - contexts: List[str]
    """
    rows = []
    for r in records:
        question = r["question"]
        answer = r["answer"]
        verses = r.get("verses", [])

        # 把所有 verse text 拼成 context 列表（每节一条）
        contexts = []
        for v in verses:
            ref = f'{v.get("book")} {v.get("chapter")}:{v.get("verse")}'
            text = v.get("text", "")
            ctx = f"[{ref}] {text}"
            contexts.append(ctx)

        # 没有 context 的就先不评估
        if not contexts:
            continue

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
        )

    df = pd.DataFrame(rows)
    return Dataset.from_pandas(df)


def main():
    # 检查 OPENAI_API_KEY 是否设置
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("   请确保 .env 文件中包含 OPENAI_API_KEY，或设置环境变量。")
        return
    
    print("📥 Loading query logs...")
    records = load_logs(max_samples=50)
    if not records:
        print("❌ No records found. 请先多问几个问题生成日志，再来跑评估。")
        return

    print(f"✅ Loaded {len(records)} raw log records.")

    ds = build_ragas_dataset(records)
    print(f"✅ Built Ragas dataset with {len(ds)} rows.")

    metrics = [
        answer_relevancy,  # 评估答案与问题的相关性（不需要 reference）
        faithfulness,      # 评估答案是否忠实于提供的上下文（不需要 reference）
        # context_precision,  # 需要 reference（ground truth）列
        # context_recall,     # 需要 reference（ground truth）列
    ]

    print("📊 Running Ragas evaluation (this will call an LLM, can take a while)...")
    result = evaluate(
        ds,
        metrics=metrics,
    )

    print("\n🎯 Evaluation Summary:")
    print(result)

    out_dir = Path(__file__).resolve().parent.parent / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = result.to_pandas()
    result_df.to_csv(out_dir / "ragas_eval_results.csv", index=False)
    print(f"\n📄 Detailed results saved to: {out_dir / 'ragas_eval_results.csv'}")


if __name__ == "__main__":
    main()