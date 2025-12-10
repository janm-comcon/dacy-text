import kenlm

class LMModel:
    def __init__(self, path):
        self.model = kenlm.Model(path)

    def score(self, sentence: str) -> float:
        raw = self.model.score(sentence, bos=True, eos=True)
        length = max(len(sentence.split()), 1)
        return raw / length
