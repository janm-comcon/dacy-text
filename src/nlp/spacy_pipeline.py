import dacy

nlp = dacy.load("small")
nlp.add_pipe("da_dacy_small_ner_fine_grained-0.1.0", config={"size": "small"})

def parse(text: str):
    return nlp(text)
