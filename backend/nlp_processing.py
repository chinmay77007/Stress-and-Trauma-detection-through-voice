import spacy

nlp = spacy.load("en_core_web_sm")


def process_text(text):
    doc = nlp(text)

    tokens = []
    lemmas = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.text)
            lemmas.append(token.lemma_)

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "sentence_count": len(list(doc.sents))
    }
