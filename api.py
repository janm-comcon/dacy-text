from fastapi import FastAPI
from pydantic import BaseModel
from nlp_pipeline import nlp, load_patterns, build_matcher, analyze

# Load patterns and construct matcher
patterns = load_patterns("patterns.json")
matcher = build_matcher(patterns)

app = FastAPI()


class TextRequest(BaseModel):
    text: str


@app.post("/analyze/")
def analyze_text(req: TextRequest):
    return analyze(req.text, matcher, patterns)


@app.post("/similarity/")
def compare_texts(txt1: TextRequest, txt2: TextRequest):
    doc1 = nlp(txt1.text)
    doc2 = nlp(txt2.text)

    return {
        "similarity": float(doc1.similarity(doc2))
    }


@app.get("/")
def root():
    return {"message": "Danish NLP API running"}
