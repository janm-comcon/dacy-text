# stdtext — Danish Grammar & Style Modeling System

This project builds a **hybrid grammar model** using:

- A **Probabilistic Dependency Grammar (PDG)** from Danish text
- A **KenLM n-gram language model**
- A **hybrid scoring function** combining grammar + style
- A **REST API** for evaluating sentences
- CLI tools for scoring, batch scoring, and PDG inspection

## Features

- Train grammar from large corpora with chunking
- Train LM models using KenLM
- Evaluate new sentences for grammatical conformity
- FastAPI service for easy integration
- spaCy-based dependency parsing

## Project Structure

See: `tree` overview in repository root.

## Training

### Train Probabilistic Grammar (PDG)
```bash
python src/training/train_pdg.py
