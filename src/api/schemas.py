from pydantic import BaseModel

class SentenceRequest(BaseModel):
    text: str

class ScoreResponse(BaseModel):
    sentence: str
    pdg_score: float
    lm_score: float
    combined_score: float
