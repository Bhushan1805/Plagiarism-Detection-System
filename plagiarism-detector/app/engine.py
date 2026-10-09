"""Detection engine: winnowing fingerprints (exact/near-exact copy)
+ TF-IDF sentence similarity (light paraphrase)."""
import hashlib
import io
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")
K = 5                # words per k-gram
W = 4                # winnowing window
PARAPHRASE_MIN = 0.60  # cosine threshold for paraphrase flag
MIN_SENT_WORDS = 6


# ---------- text extraction ----------
def extract_text(filename: str, data: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "txt":
        return data.decode("utf-8", "ignore")
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == "docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")


# ---------- preprocessing ----------
def tokenize(text: str):
    """[(word, start_char, end_char)]"""
    return [(m.group().lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


def winnow(tokens):
    """Return list of (hash, token_position) fingerprints."""
    words = [t[0] for t in tokens]
    if len(words) < K:
        return []
    hashes = [_h(" ".join(words[i:i + K])) for i in range(len(words) - K + 1)]
    fps, last = [], -1
    for i in range(max(1, len(hashes) - W + 1)):
        win = hashes[i:i + W]
        m = min(win)
        j = i + (len(win) - 1 - win[::-1].index(m))  # rightmost minimum
        if j != last:
            fps.append((m, j))
            last = j
    return fps


def sentences(text: str):
    out = []
    for m in SENT_RE.finditer(text):
        s = m.group().strip()
        if len(TOKEN_RE.findall(s)) >= MIN_SENT_WORDS:
            out.append((s, m.start(), m.end()))
    return out


# ---------- corpus ----------
class Corpus:
    def __init__(self):
        self.docs = {}   # id -> dict

    def add(self, doc_id, title, text):
        toks = tokenize(text)
        fp_map = {}
        for h, pos in winnow(toks):
            fp_map.setdefault(h, []).append(pos)
        self.docs[doc_id] = {
            "title": title, "text": text, "tokens": toks,
            "fps": fp_map, "sents": sentences(text),
        }

    def remove(self, doc_id):
        self.docs.pop(doc_id, None)


# ---------- checking ----------
def _runs(sorted_positions):
    runs, start, prev = [], None, None
    for p in sorted_positions:
        if start is None:
            start = prev = p
        elif p == prev + 1:
            prev = p
        else:
            runs.append((start, prev))
            start = prev = p
    if start is not None:
        runs.append((start, prev))
    return runs


def check(text: str, corpus: Corpus):
    tokens = tokenize(text)
    total = len(tokens)
    if total == 0:
        return {"score": 0, "total_words": 0, "matches": [], "sources": []}

    sub_fps = winnow(tokens)
    covered = set()          # token indices flagged (any source)
    matches = []
    per_source = {}          # doc_id -> set(token idx)

    # 1) exact / near-exact: shared fingerprints
    for doc_id, doc in corpus.docs.items():
        sub_to_doc = {}
        for h, spos in sub_fps:
            if h in doc["fps"]:
                dpos = doc["fps"][h][0]
                for off in range(K):
                    if spos + off < total:
                        sub_to_doc.setdefault(spos + off, dpos + off)
        if not sub_to_doc:
            continue
        for a, b in _runs(sorted(sub_to_doc)):
            if b - a + 1 < K:
                continue
            d_lo = min(sub_to_doc[i] for i in range(a, b + 1))
            d_hi = min(max(sub_to_doc[i] for i in range(a, b + 1)), len(doc["tokens"]) - 1)
            snippet = doc["text"][doc["tokens"][d_lo][1]:doc["tokens"][d_hi][2]]
            matches.append({
                "start": tokens[a][1], "end": tokens[b][2],
                "type": "exact", "score": 1.0,
                "source_id": doc_id, "source_title": doc["title"],
                "source_text": snippet,
            })
            idx = set(range(a, b + 1))
            covered |= idx
            per_source.setdefault(doc_id, set()).update(idx)

    # 2) paraphrase: TF-IDF cosine at sentence level
    sub_sents = sentences(text)
    corp_sents = [(did, s) for did, d in corpus.docs.items() for s in d["sents"]]
    if sub_sents and corp_sents:
        corpus_text = [s[0] for _, s in corp_sents]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        vec.fit(corpus_text + [s[0] for s in sub_sents])
        sims = cosine_similarity(vec.transform([s[0] for s in sub_sents]),
                                 vec.transform(corpus_text))
        for i, (s, st, en) in enumerate(sub_sents):
            j = int(sims[i].argmax())
            score = float(sims[i][j])
            if score < PARAPHRASE_MIN:
                continue
            idxs = {k for k, t in enumerate(tokens) if t[1] >= st and t[2] <= en}
            if not idxs or len(idxs & covered) / len(idxs) > 0.5:
                continue  # already explained by an exact match
            did, src = corp_sents[j]
            matches.append({
                "start": st, "end": en, "type": "paraphrase",
                "score": round(score, 2),
                "source_id": did, "source_title": corpus.docs[did]["title"],
                "source_text": src[0],
            })
            covered |= idxs
            per_source.setdefault(did, set()).update(idxs)

    matches.sort(key=lambda m: m["start"])
    sources = sorted(
        ({"id": d, "title": corpus.docs[d]["title"],
          "percent": round(100 * len(ix) / total, 1)} for d, ix in per_source.items()),
        key=lambda s: -s["percent"])
    return {
        "score": round(100 * len(covered) / total, 1),
        "total_words": total,
        "matches": matches,
        "sources": sources,
    }
