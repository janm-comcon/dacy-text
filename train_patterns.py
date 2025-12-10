import json
import glob
from collections import Counter
import spacy

nlp = spacy.load("da_core_news_sm")

# ----------------------------------------------------
# Stream a large text file in chunks
# ----------------------------------------------------
def load_text_chunks(filepath, chunk_size=20000):
    """
    Reads a large text file piece-by-piece.
    chunk_size is the number of characters processed at once.
    """
    for file in glob.glob(filepath):
        with open(file, "r", encoding="utf-8") as f:
            buffer = f.read(chunk_size)
            while buffer:
                yield buffer
                buffer = f.read(chunk_size)


# ----------------------------------------------------
# Extract hybrid dependency grammar patterns
# ----------------------------------------------------
def extract_hybrid_patterns(doc):
    for tok in doc:
        yield (
            tok.lemma_,
            tok.pos_,
            tok.dep_,
            tok.head.pos_
        )


# ----------------------------------------------------
# Train grammar patterns from a large file, chunked
# ----------------------------------------------------
def train_from_large_textfile(filepath, n_patterns=200, chunk_size=20000):
    counter = Counter()

    for chunk in load_text_chunks(filepath, chunk_size=chunk_size):

        # DaCy processes each chunk
        doc = nlp(chunk)

        # Extract grammar patterns
        for pattern in extract_hybrid_patterns(doc):
            counter[pattern] += 1

    # Top-N most frequent grammar structures
    most_common = counter.most_common(n_patterns)

    json_patterns = [
        {
            "lemma": lemma,
            "pos": pos,
            "dep": dep,
            "head_pos": head_pos,
            "count": count
        }
        for (lemma, pos, dep, head_pos), count in most_common
    ]

    return json_patterns


# ----------------------------------------------------
# Save results
# ----------------------------------------------------
def save_patterns(patterns, outfile="patterns.json"):
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(patterns)} patterns → {outfile}")


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    # Change this to point at your big text file:
    infile = "C:/temp/train/*.txt"

    patterns = train_from_large_textfile(
        filepath=infile,
        chunk_size=20000,     # ~20KB per chunk
        n_patterns=200
    )

    save_patterns(patterns, "patterns.json")
    print("Chunk-based pattern training complete.")
