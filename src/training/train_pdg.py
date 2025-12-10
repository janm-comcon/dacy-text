import json
from collections import Counter, defaultdict
import spacy

# Load Danish spaCy model (install via: python -m spacy download da_core_news_sm)
nlp = spacy.load("da_core_news_sm")

def read_in_chunks(filepath, chunk_size=500):
    chunk = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def train_pdg(corpus_path, chunk_size=500):
    # counts of (head_pos, dep, child_pos) triples
    triple_counts = Counter()
    # counts of (head_pos, dep) for normalization
    head_dep_counts = Counter()

    for chunk in read_in_chunks(corpus_path, chunk_size):
        for doc in nlp.pipe(chunk):
            for tok in doc:
                head_pos = tok.head.pos_
                dep = tok.dep_
                child_pos = tok.pos_

                triple = (head_pos, dep, child_pos)
                triple_counts[triple] += 1
                head_dep_counts[(head_pos, dep)] += 1

    # convert to probabilities
    probs = {}
    for (head_pos, dep, child_pos), count in triple_counts.items():
        denom = head_dep_counts[(head_pos, dep)]
        probs.setdefault(head_pos, {}).setdefault(dep, {})[child_pos] = count / denom

    return {
        "triple_counts": {f"{h}|{d}|{c}": cnt for (h, d, c), cnt in triple_counts.items()},
        "head_dep_counts": {f"{h}|{d}": cnt for (h, d), cnt in head_dep_counts.items()},
        "probs": probs,
    }

def save_pdg(model, outfile="grammar_stats.json"):
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    pdg = train_pdg("large_training_text.txt")
    save_pdg(pdg, "grammar_stats.json")
    print("PDG training done → grammar_stats.json")
