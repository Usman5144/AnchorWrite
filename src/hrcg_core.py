import re


_WS_RE = re.compile(r"\s+")


def clean_text(text):
    if text is None:
        return ""
    return _WS_RE.sub(" ", str(text)).strip()


def word_count(text):
    text = clean_text(text)
    if not text:
        return 0
    return len(text.split())


def _cross_entropy(model, text):
    text = clean_text(text)
    n = max(1, word_count(text))
    return -float(model.score(text, bos=True, eos=True)) / n


def score_document(model2, model4, text):
    text = clean_text(text)

    h2 = _cross_entropy(model2, text)
    h4 = _cross_entropy(model4, text)

    denom = abs(h2)
    if denom < 1e-12:
        a24 = 0.0
    else:
        a24 = (h2 - h4) / denom

    return {
        "H2": float(h2),
        "H4": float(h4),
        "A24": float(a24),
        "H2_minus_H4": float(h2 - h4),
        "word_count": int(word_count(text)),
    }
