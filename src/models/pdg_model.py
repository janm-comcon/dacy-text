import json
import math
import spacy

nlp = spacy.load("da_core_news_sm")

class PDGModel:
    def __init__(self, path, smoothing=1e-6):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.probs = data["probs"]
        self.smoothing = smoothing

    def score(self, sentence: str) -> float:
        doc = nlp(sentence)
        logps = []
        for tok in doc:
            p = (
                self.probs
                .get(tok.head.pos_, {})
                .get(tok.dep_, {})
                .get(tok.pos_, self.smoothing)
            )
            logps.append(math.log(p))
        return sum(logps) / max(len(logps), 1)
