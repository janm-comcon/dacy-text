"""
Extract verb lemmas from the domain corpus and update the lemma list used by the spellchecker.

Usage:
    python -m src.training.extract_lemmas --input data/lm/lm_corpus.txt --output data/lm/lemmas.json --top-k 300

This preserves existing lemma entries (including objects) and adds missing verb lemmas under the "actions" key.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Ensure project root on sys.path so `python src/api/main.py` works
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.spacy_pipeline import nlp


def extract_action_lemmas(corpus_path: Path, top_k: int) -> List[str]:
    """Return the most frequent verb lemmas from the corpus."""
    counts: Counter[str] = Counter()
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for tok in nlp(line):
            if tok.pos_ == "VERB":
                lemma = tok.lemma_.lower()
                counts[lemma] += 1
    return [lemma for lemma, _ in counts.most_common(top_k)]


def load_existing_lemmas(path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not path.exists():
        return {"actions": {}, "objects": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"actions": {}, "objects": {}}
    return {
        "actions": data.get("actions", {}) or {},
        "objects": data.get("objects", {}) or {},
    }


def update_actions(existing: Dict[str, Dict[str, List[str]]], new_lemmas: List[str]) -> Dict[str, Dict[str, List[str]]]:
    actions = existing.get("actions", {}) or {}
    for lemma in new_lemmas:
        actions.setdefault(lemma, [])
    existing["actions"] = actions
    return existing


def main():
    parser = argparse.ArgumentParser(description="Extract verb lemmas from corpus into lemmas.json")
    parser.add_argument("--input", required=True, help="Path to domain corpus (cleaned, e.g., data/lm/lm_corpus.txt)")
    parser.add_argument("--output", required=True, help="Path to lemmas.json to write/update")
    parser.add_argument("--top-k", type=int, default=300, help="How many verb lemmas to keep (by frequency)")
    args = parser.parse_args()

    corpus_path = Path(args.input)
    out_path = Path(args.output)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    lemmas = extract_action_lemmas(corpus_path, top_k=args.top_k)
    data = load_existing_lemmas(out_path)
    data = update_actions(data, lemmas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(lemmas)} action lemmas to {out_path}")


if __name__ == "__main__":
    main()
