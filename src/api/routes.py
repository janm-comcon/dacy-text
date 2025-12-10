from fastapi import APIRouter
from .schemas import SentenceRequest, ScoreResponse
from ..models.hybrid_model import HybridModel

router = APIRouter()

model = HybridModel(
    pdg_path="data/pdg/grammar_stats.json",
    lm_path="data/lm/my_corpus.bin",
    alpha=0.5
)

@router.post("/score", response_model=ScoreResponse)
def score(req: SentenceRequest):
    return model.score(req.text)
