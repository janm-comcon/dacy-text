import re
from pathlib import Path

# -------------------------
# PATTERNS
# -------------------------

# WORK ORDER IDs, e.g. "DA 113/539 | DA 123-456 | DA "
WORK_ORDER_PATTERN = re.compile(r"\bD?A?\s*\d+[\/\-]\d+|DA\s*\b", re.IGNORECASE)

# VEDR. (task reference)
ADDRESS_TAG = re.compile(r"^VEDR\.", re.IGNORECASE)

# Status/ending words typical in your domain
STATUS_WORDS = {"afprøvet", "fundet i orden", "fundet iorden", "justeret"}

# -------------------------
# HELPERS
# -------------------------

def remove_work_orders(text: str) -> str:
    """Remove DA 123/456 style codes."""
    return WORK_ORDER_PATTERN.sub("", text)


def clean_spaces(t: str) -> str:
    """Normalize spacing."""
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\\ns*", "", t)
    return t


def normalize_case(t: str) -> str:
    """
    Lowercase domain text; this improves LM/PDG quality.
    Work codes are removed entirely, so no exceptions needed.
    """
    return t.lower()


def segment_line(line: str):
    """
    Domain-aware segmentation for electrical/maintenance work orders.

    Rules:
    - Remove work order IDs.
    - Split on '.' if present, but keep short fragments attached.
    - Lines starting with 'VEDR.' begin new segments.
    - Merge overly short fragments into previous segment.
    """
    original = line.strip()
    if not original:
        return []

    # Remove work order codes like "DA 113/539"
    original = remove_work_orders(original)

    # Split by periods, but preserve non-empty segments
    parts = re.split(r"\.\s*", original)
    parts = [clean_spaces(p) for p in parts if p.strip()]

    merged = []
    buffer = ""

    for part in parts:

        # New action begins with "VEDR."
        if ADDRESS_TAG.match(part):
            if buffer:
                merged.append(buffer)
            buffer = part
            continue

        # Very short fragments belong to previous part (e.g., "Afprøvet")
        if len(part.split()) <= 2:
            if buffer:
                buffer += " " + part
            else:
                buffer = part
            continue

        # Normal merging case
        if buffer:
            buffer += " " + part
        else:
            buffer = part

    if buffer:
        merged.append(buffer)

    return merged


# -------------------------
# MAIN CLEANER
# -------------------------

def prepare_corpus(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as fin, \
         output_file.open("w", encoding="utf-8") as fout:

        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            segments = segment_line(raw_line)

            for seg in segments:
                seg = clean_spaces(seg)
                seg = normalize_case(seg)

                # Filter out too-short lines
                if len(seg.split()) < 3:
                    continue

                fout.write(seg + "\n")

    print(f"Domain-aware LM corpus written → {output_file}")

if __name__ == "__main__":
    prepare_corpus(
        input_file="data/raw/large_training_text.txt",
        output_file="data/lm/lm_corpus.txt")