import math
from .pdg_model import PDGModel
from .lm_model import LMModel

class HybridModel:
    def __init__(self, pdg_path, lm_path, alpha=0.5):
        self.pdg = PDGModel(pdg_path)
        self.lm = LMModel(lm_path)
        self.alpha = alpha

    def score(self, sentence: str) -> dict:
        pdg_s = self.pdg.score(sentence)
        lm_s = self.lm.score(sentence)

        combined = self.alpha * pdg_s + (1 - self.alpha) * lm_s

        return {
            "sentence": sentence,
            "pdg_score": pdg_s,
            "lm_score": lm_s,
            "combined_score": combined
        }
