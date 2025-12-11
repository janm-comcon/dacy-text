from pydantic import BaseModel

class Correction(BaseModel):
    original: str
    suggestion: str
    position: int


class SentenceRequest(BaseModel):
    text: str
    autocorrect: bool = True


class ScoreResponse(BaseModel):
    sentence: str
    corrected_sentence: str
    corrections: list[Correction]
    pdg_score: float
    lm_score: float
    combined_score: float
