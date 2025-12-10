import spacy

nlp = spacy.load("da_core_news_sm")

def parse(text: str):
    return nlp(text)
