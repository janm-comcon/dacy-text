from fastapi import APIRouter

from schemas import SentenceRequest, ScoreResponse
from src.models.hybrid_model import HybridModel

router = APIRouter()

model = HybridModel(
    pdg_path="data/pdg/grammar_stats.json",
    lm_path="data/lm/my_corpus.bin",
    alpha=0.5,
    vocab_path="data/lm/lm_corpus.txt",
    correction_cutoff=0.82,
)

@router.post("/score", response_model=ScoreResponse)
def score(req: SentenceRequest):
    return model.score(req.text, autocorrect=req.autocorrect)
