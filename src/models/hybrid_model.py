from src.models.pdg_model import PDGModel
from src.models.lm_model import LMModel
from src.models.spellchecker import SpellChecker


class HybridModel:
    def __init__(self, pdg_path, lm_path, alpha=0.5, vocab_path=None, correction_cutoff: float = 0.8):
        self.pdg = PDGModel(pdg_path)
        self.lm = LMModel(lm_path)
        self.alpha = alpha
        self.spellchecker = SpellChecker(vocab_path, cutoff=correction_cutoff) if vocab_path else None

    def score(self, sentence: str, autocorrect: bool = True) -> dict:
        corrections = {"corrected_sentence": sentence, "corrections": []}
        text_for_scoring = sentence

        if autocorrect and self.spellchecker:
            corrections = self.spellchecker.correct(sentence)
            text_for_scoring = corrections["corrected_sentence"]

        pdg_s = self.pdg.score(text_for_scoring)
        lm_s = self.lm.score(text_for_scoring)

        combined = self.alpha * pdg_s + (1 - self.alpha) * lm_s

        return {
            "sentence": sentence,
            "corrected_sentence": corrections["corrected_sentence"],
            "corrections": corrections["corrections"],
            "pdg_score": pdg_s,
            "lm_score": lm_s,
            "combined_score": combined,
        }
