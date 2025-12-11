import difflib
from pathlib import Path
from typing import List, Dict

from src.nlp.spacy_pipeline import nlp


class SpellChecker:
    def __init__(self, corpus_path: str, cutoff: float = 0.8):
        self.corpus_path = Path(corpus_path)
        self.cutoff = cutoff
        self.vocab = self._load_vocab()

    def _load_vocab(self) -> set:
        vocab = set()
        if not self.corpus_path.exists():
            return vocab

        with self.corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                for token in line.strip().split():
                    vocab.add(token.lower())
        return vocab

    def suggest(self, token: str) -> str:
        lower = token.lower()
        if lower in self.vocab or not token.isalpha():
            return token

        matches = difflib.get_close_matches(lower, self.vocab, n=1, cutoff=self.cutoff)
        if not matches:
            return token

        suggestion = matches[0]
        if token.istitle():
            suggestion = suggestion.capitalize()
        elif token.isupper():
            suggestion = suggestion.upper()
        return suggestion

    def correct(self, sentence: str) -> Dict[str, object]:
        doc = nlp(sentence)

        corrected_tokens: List[str] = []
        corrections: List[Dict[str, object]] = []

        for tok in doc:
            suggestion = self.suggest(tok.text)
            corrected_tokens.append(suggestion + tok.whitespace_)

            if suggestion != tok.text:
                corrections.append(
                    {
                        "original": tok.text,
                        "suggestion": suggestion,
                        "position": tok.i,
                    }
                )

        corrected_sentence = "".join(corrected_tokens)
        return {
            "corrected_sentence": corrected_sentence,
            "corrections": corrections,
        }
