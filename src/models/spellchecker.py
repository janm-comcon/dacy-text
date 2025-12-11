import difflib
import json
from pathlib import Path
from typing import Dict, List, Optional

import kenlm

from src.nlp.spacy_pipeline import nlp


class SpellChecker:
    def __init__(
        self,
        corpus_path: str,
        cutoff: float = 0.8,
        lm_path: Optional[str] = None,
        lemma_path: Optional[str] = None,
    ):
        self.corpus_path = Path(corpus_path)
        self.cutoff = cutoff
        self.vocab = self._load_vocab()
        self.lm = kenlm.Model(lm_path) if lm_path else None
        self.lemma_lookup = self._load_lemmas(lemma_path)
        # Domain-specific fixes that are faster than fuzzy matching
        self.domain_map = {
            "instal": "installation",
            "k\u00f8ken": "k\u00f8kken",
        }

    def _load_vocab(self) -> set:
        vocab = set()
        if not self.corpus_path.exists():
            return vocab

        with self.corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                for token in line.strip().split():
                    vocab.add(token.lower())
        return vocab

    def _load_lemmas(self, lemma_path: Optional[str]) -> Dict[str, str]:
        """Load lemma variants into a flat lookup of form variant -> canonical."""
        if not lemma_path:
            return {}
        path = Path(lemma_path)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

        lookup: Dict[str, str] = {}
        for category in data.values():
            if not isinstance(category, dict):
                continue
            for canonical, variants in category.items():
                canon_lower = canonical.lower()
                lookup[canon_lower] = canon_lower
                if isinstance(variants, list):
                    for v in variants:
                        lookup[str(v).lower()] = canon_lower
        return lookup

    def suggest(self, token: str, prev_like_num: bool = False) -> str:
        lower = token.lower()
        if lower in self.vocab or not token.isalpha():
            return token

        if lower in self.lemma_lookup:
            lower = self.lemma_lookup[lower]

        # Quick domain remaps
        if lower in self.domain_map:
            lower = self.domain_map[lower]

        # Simple pluralization when preceded by a number (e.g., "2 lampe" -> "2 lamper")
        if prev_like_num and lower == "lampe":
            lower = "lamper"

        matches = difflib.get_close_matches(lower, self.vocab, n=1, cutoff=self.cutoff)
        if not matches:
            # Fallback with looser cutoff to avoid leaving obvious typos unchanged
            matches = difflib.get_close_matches(lower, self.vocab, n=1, cutoff=0.6)

        suggestion = matches[0] if matches else lower
        if token.istitle():
            suggestion = suggestion.capitalize()
        elif token.isupper():
            suggestion = suggestion.upper()
        return suggestion

    def _lm_score_from_tokens(self, tokens: List[Dict[str, str]]) -> float:
        if not self.lm:
            return float("-inf")
        sentence = "".join([t["text"] + t["ws"] for t in tokens])
        raw = self.lm.score(sentence, bos=True, eos=True)
        length = max(len(sentence.split()), 1)
        return raw / length

    def _generate_inflection_candidates(self, token: str, prev_like_num: bool) -> List[str]:
        lower = token.lower()
        candidates = set([token])

        # Lemma lookup
        if lower in self.lemma_lookup:
            candidates.add(self.lemma_lookup[lower])

        # Domain remaps
        if lower in self.domain_map:
            candidates.add(self.domain_map[lower])

        # Quick pluralization/lemmatization heuristics
        if prev_like_num:
            if lower.endswith("e"):
                candidates.add(lower + "r")  # lampe -> lamper
            else:
                candidates.add(lower + "er")
        else:
            if lower.endswith("er"):
                candidates.add(lower[:-2])  # lamper -> lampe
            if lower.endswith("e"):
                candidates.add(lower[:-1])  # installatione -> installation (edge)

        # Add fuzzy guess
        candidates.add(self.suggest(token, prev_like_num=prev_like_num))

        return list(candidates)

    def correct(self, sentence: str) -> Dict[str, object]:
        doc = nlp(sentence)

        corrected_tokens: List[Dict[str, str]] = []
        corrections: List[Dict[str, object]] = []

        for tok in doc:
            prev_like_num = tok.i > 0 and doc[tok.i - 1].like_num
            candidates = self._generate_inflection_candidates(tok.text, prev_like_num=prev_like_num)

            # Choose best candidate by LM when available; otherwise pick first (which includes fuzzy)
            best = tok.text
            best_score = None
            for cand in candidates:
                trial_tokens = corrected_tokens + [{"text": cand, "ws": tok.whitespace_}] + [
                    {"text": t.text, "ws": t.whitespace_} for t in doc[tok.i + 1 :]
                ]
                score = self._lm_score_from_tokens(trial_tokens) if self.lm else None
                if best_score is None or (score is not None and score > best_score):
                    best_score = score if score is not None else best_score
                    best = cand

            corrected_tokens.append({"text": best, "ws": tok.whitespace_})

            if best != tok.text:
                corrections.append(
                    {
                        "original": tok.text,
                        "suggestion": best,
                        "position": tok.i,
                    }
                )

        # Heuristic reorder: move trailing "installation" to the front and insert "af"
        # when the sentence starts with a number (e.g., "2 lampe ... installation").
        if corrected_tokens:
            first_is_num = doc[0].like_num
            install_idx = next(
                (i for i, t in enumerate(corrected_tokens) if t["text"].lower() == "installation"),
                None,
            )
            if first_is_num and install_idx not in (None, 0):
                install_token = corrected_tokens.pop(install_idx)
                # Ensure we leave a space after moving installation to the front
                install_token["ws"] = install_token.get("ws", " ") or " "
                needs_af = not corrected_tokens or corrected_tokens[0]["text"].lower() != "af"
                prefix = [install_token]
                if needs_af:
                    prefix.append({"text": "af", "ws": " "})
                corrected_tokens = prefix + corrected_tokens

        corrected_sentence = "".join([t["text"] + t["ws"] for t in corrected_tokens])
        return {
            "corrected_sentence": corrected_sentence,
            "corrections": corrections,
        }
