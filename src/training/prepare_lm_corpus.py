from pathlib import Path
import re

def clean_segment(seg: str) -> str:
    # Normalize spacing and uppercase (your corpus is mostly uppercase)
    seg = seg.strip()
    seg = re.sub(r"\s+", " ", seg)
    return seg

def prepare_corpus(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as fin, \
         output_file.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            # Split line by dots into meaningful segments
            parts = re.split(r"\.\s*", line)

            for part in parts:
                part = clean_segment(part)
                if len(part) < 3:
                    continue
                fout.write(part + "\n")

    print(f"LM corpus created → {output_file}")

if __name__ == "__main__":
    prepare_corpus(
        input_file="data/raw/large_training_text.txt",
        output_file="data/lm/lm_corpus.txt",
        chunk_lines=500
    )