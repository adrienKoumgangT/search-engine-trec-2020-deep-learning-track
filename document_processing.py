from __future__ import annotations

from pathlib import Path
from typing import Iterator

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.snowball import SnowballStemmer

_stemmer = SnowballStemmer("english")
_stem_cache: dict[str, str] = {}

# Source: NLTK English stopword list (https://www.nltk.org/nltk_data/)
try:
    DEFAULT_STOPWORDS: set[str] = set(nltk_stopwords.words("english"))
except LookupError:
    import nltk

    nltk.download("stopwords", quiet=True)
    DEFAULT_STOPWORDS = set(nltk_stopwords.words("english"))


def process_text(
    text: str,
    use_stopwords: bool = True,
    use_stemming: bool = True,
    stopwords: set[str] | None = None,
) -> list[str]:
    text = case_law_normalization(text)
    text = non_alphanumeric_removal(text)
    tokens = tokenization(text)
    if use_stopwords:
        tokens = stopword_removal(tokens, stopwords if stopwords is not None else DEFAULT_STOPWORDS)
    if use_stemming:
        tokens = stemming(tokens)
    return tokens


def case_law_normalization(text: str) -> str:
    return text.lower()


def non_alphanumeric_removal(text: str) -> str:
    return "".join(c if c.isalnum() else " " for c in text)


def tokenization(text: str) -> list[str]:
    return text.split()


def stopword_removal(tokens: list[str], stopwords: set[str]) -> list[str]:
    return [token for token in tokens if token not in stopwords]


def stemming(tokens: list[str]) -> list[str]:
    """Apply Snowball English stemming to each token, reusing cached stems for distinct tokens."""
    stemmed: list[str] = []
    for token in tokens:
        stem = _stem_cache.get(token)
        if stem is None:
            stem = _stemmer.stem(token)
            _stem_cache[token] = stem
        stemmed.append(stem)
    return stemmed


def document_length(tokens: list[str]) -> int:
    """|d| = number of tokens after processing."""
    return len(tokens)


def parse_collection_line(line: str) -> tuple[int, str] | None:
    """Parse one collection line. Returns (pid, text) or None if malformed."""
    text_line = line.rstrip("\r\n")
    parts = text_line.split("\t", 1)
    if len(parts) != 2:
        return None
    pid_str, text = parts
    try:
        pid = int(pid_str)
    except ValueError:
        return None
    return pid, text


def iter_collection(
    path: str | Path,
    *,
    use_stopwords: bool = True,
    use_stemming: bool = True,
    stopwords: set[str] | None = None,
    max_lines: int | None = None,
    report: bool = True,
) -> Iterator[tuple[int, int, list[str], int]]:
    """
    Stream collection.tsv as (docid, pid, tokens, length).

    Decodes UTF-8 with errors="replace". Skips lines without a tab or with a
    non-integer pid. Keeps empty-after-processing passages (length 0).
    When the iterator is exhausted, prints skipped / empty counts if report=True.
    """
    skipped = 0
    empty = 0
    docid = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            if max_lines is not None and line_no >= max_lines:
                break

            parsed = parse_collection_line(line)
            if parsed is None:
                skipped += 1
                continue

            pid, text = parsed
            tokens = process_text(
                text,
                use_stopwords=use_stopwords,
                use_stemming=use_stemming,
                stopwords=stopwords,
            )
            length = document_length(tokens)
            if length == 0:
                empty += 1

            yield docid, pid, tokens, length
            docid += 1

    if report:
        print(f"Kept passages   : {docid:,}")
        print(f"Skipped lines   : {skipped:,}")
        print(f"Empty passages  : {empty:,}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_process_text() -> None:
    """Sanity checks with the same examples used in the notebook."""
    examples = [
        "The Quick, Brown Fox!",
        "café — naïve résumé",
        "!!!",
        "",
    ]
    for s in examples:
        toks = process_text(s)
        print(f"{s!r:30s} -> {toks}  |d|={document_length(toks)}")
        assert isinstance(toks, list)
        assert all(isinstance(t, str) for t in toks)
        assert document_length(toks) == len(toks)

    # no flags: punctuation / non-ASCII kept as tokens (only case-folded)
    toks = process_text("café — naïve résumé", use_stopwords=False, use_stemming=False)
    assert "café" in toks

    # flags on: lower-cased, non-alnum stripped, stopwords out, stems cached
    toks = process_text("The Quick, Brown Fox!")
    assert all(t == t.lower() for t in toks)
    assert all(t.isalnum() for t in toks)
    assert "the" not in toks
    assert stemming(["running"]) == stemming(["running"])  # cache hit

    # malformed line handling
    assert parse_collection_line("123\tsome text") == (123, "some text")
    assert parse_collection_line("no tab here") is None
    assert parse_collection_line("abc\ttext") is None

    print("All document_processing tests passed.")


if __name__ == "__main__":
    test_process_text()
