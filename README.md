# stdtext — Danish Grammar & Style Modeling System

`stdtext` builds a hybrid Danish grammar/style model that blends a probabilistic dependency grammar (PDG) trained with spaCy and a KenLM n-gram language model. The package exposes both a FastAPI service and a small CLI for scoring sentences with grammar- and style-aware metrics.

## Components
- **PDG model** generated from dependency parses of a Danish corpus (`data/lm/lm_corpus.txt` by default).
- **KenLM language model** stored as `data/lm/my_corpus.bin` with the source ARPA file in `data/lm/my_corpus.arpa`.
- **Hybrid scorer** that mixes PDG and LM scores (configurable `alpha` weight) and powers both the API and CLI helpers.

## Setup
1. Create and activate a virtual environment (Python 3.10+).
2. Install dependencies: `pip install -r requirements.txt`.
3. Download the Danish spaCy model: `python -m spacy download da_core_news_sm` (or run `scripts/setup_env.cmd`).

## Training
### Probabilistic Dependency Grammar
Run the training script to parse the corpus and write PDG statistics:
```bash
python src/training/train_pdg.py
# outputs data/pdg/grammar_stats.json (create data/pdg first if it is missing)
```
Update the corpus path inside `train_pdg.py` if you want to use a different dataset.

### KenLM n-gram model
Rebuild the language model from the corpus with KenLM CLI tools:
```bash
lmplz -o 5 < data/lm/lm_corpus.txt > data/lm/my_corpus.arpa
build_binary data/lm/my_corpus.arpa data/lm/my_corpus.bin
```
> **Note:** Build and binarize the KenLM model on a Linux environment. The binary format is platform-specific, so generating it on Linux avoids compatibility issues when the service loads `data/lm/my_corpus.bin`.

## Using the models
### FastAPI service
Start the API (expects `data/pdg/grammar_stats.json` and `data/lm/my_corpus.bin` to exist):
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
Score text via HTTP:
```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"text": "Jeg har en stor hund som elsker at lege."}'
```

### CLI quick check
The CLI helper in `src/cli/scor_sentence.py` demonstrates hybrid scoring for a single sentence. Ensure `grammar_stats.json` and `my_corpus.bin` are available in your working directory (or edit the paths), then run:
```bash
python -m src.cli.scor_sentence
```
