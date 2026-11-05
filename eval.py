#!/usr/bin/env python3
"""
Score FTC GPT against eval.csv. Needs `ollama serve` running.

Usage:
    python eval.py [--questions eval.csv] [--out eval_results.csv] [--model NAME]

Prints retrieval and keyword scores, and writes every answer to the output CSV.
Keyword hits are only a screen: read the answers and fill in `correct` by hand.
"""

import argparse
import csv

from ai import DEFAULT_ANSWER_MODEL, ChunkStore, answer, default_chunks


def score(row: dict, result: dict) -> dict:
    sources = result.get("sources", [])
    text = result.get("answer", "")
    wanted = row["expected_source"].strip()
    keywords = [k.strip() for k in row["must_include"].split(";") if k.strip()]
    return {
        "retrieved": any(wanted.lower() in s.lower() for s in sources) if wanted else "",
        "keywords": all(k.lower() in text.lower() for k in keywords) if keywords else "",
        "sources": "; ".join(sources),
        "answer": text,
        "correct": "",
    }


def rate(rows: list, field: str) -> str:
    scored = [r[field] for r in rows if r[field] != ""]
    return f"{sum(scored)}/{len(scored)}" if scored else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="eval.csv")
    parser.add_argument("--out", default="eval_results.csv")
    parser.add_argument("--chunks", default=default_chunks())
    parser.add_argument("--model", default=DEFAULT_ANSWER_MODEL)
    args = parser.parse_args()

    store = ChunkStore(args.chunks)
    with open(args.questions, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    rows = []
    for i, row in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {row['question']}")
        try:
            result = answer([{"role": "user", "content": row["question"]}], store, args.model)
        except Exception as e:
            result = {"answer": f"ERROR: {e}", "sources": []}
        rows.append({**row, **score(row, result)})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRight source retrieved: {rate(rows, 'retrieved')}")
    print(f"Keywords in answer:     {rate(rows, 'keywords')}")
    print(f"FRC manual retrieved:   {sum('2026GameManual' in r['sources'] for r in rows)}/{len(rows)}")
    print(f"TU02 retrieved:         {sum('TU02' in r['sources'] for r in rows)}/{len(rows)}")
    print(f"Wrote {args.out}. Fill in `correct` by hand.")


if __name__ == "__main__":
    main()
