import json
import math
import spacy
import kenlm

# Load spaCy model
nlp = spacy.load("da_core_news_sm")

# Load PDG
with open("grammar_stats.json", "r", encoding="utf-8") as f:
    PDG = json.load(f)

PDG_PROBS = PDG["probs"]  # nested dict: head_pos -> dep -> child_pos -> prob

# Load KenLM LM
LM = kenlm.Model("my_corpus.bin")  # path to your LM

def pdg_score(sentence: str) -> float:
    """
    Compute average log-probability of its dependencies under PDG.
    """
    doc = nlp(sentence)
    logps = []
    for tok in doc:
        head_pos = tok.head.pos_
        dep = tok.dep_
        child_pos = tok.pos_

        # Look up probability, with small smoothing
        p = (
            PDG_PROBS
            .get(head_pos, {})
            .get(dep, {})
            .get(child_pos, 1e-6)  # floor prob
        )
        logps.append(math.log(p))

    if not logps:
        return float("-inf")
    return sum(logps) / len(logps)  # average log-prob per dependency

def lm_score(sentence: str) -> float:
    """
    KenLM's average log-prob per token.
    """
    # KenLM score: log10 prob; we can normalize by length
    raw = LM.score(sentence, bos=True, eos=True)  # log10
    length = max(len(sentence.split()), 1)
    return raw / length

def hybrid_score(sentence: str, alpha: float = 0.5) -> dict:
    """
    Combine PDG and LM scores.
    alpha = weight on PDG; (1 - alpha) on LM.
    """
    s_pdg = pdg_score(sentence)
    s_lm = lm_score(sentence)
    combined = alpha * s_pdg + (1 - alpha) * s_lm

    return {
        "sentence": sentence,
        "pdg_score": s_pdg,
        "lm_score": s_lm,
        "combined_score": combined,
    }

if __name__ == "__main__":
    s = "Jeg har en stor hund som elsker at lege."
    res = hybrid_score(s)
    print(res)
