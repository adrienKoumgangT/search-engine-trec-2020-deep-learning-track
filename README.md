# Search Engine for The MS MARCO passage collection (TREC Deep Learning 2020)

Use **Python 3.13.18**, matching the current Google Colab runtime. Develop and test locally against that version so behaviour matches Colab.

## Environment setup (`.venv`)

From the repository root:

```bash
# Confirm the interpreter (must be 3.13.18)
python3.13 --version   # → Python 3.13.18

# Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

`.venv` is listed in `.gitignore` and must not be committed.

## Document processing (`document_processing.py`)

C1 text pipeline: lower-case, replace non-alphanumeric characters with spaces, whitespace tokenisation, optional stop-word removal (NLTK English), optional Snowball English stemming with a per-token cache. Also streams `collection.tsv` with UTF-8 `errors="replace"`, skips malformed lines, and keeps empty-after-processing passages.

### Run the tests

With `.venv` activated, from the repository root:

```bash
python document_processing.py
```

This executes `test_process_text()`: prints a few processed examples, asserts case / punctuation / stopword / stemming / malformed-line behaviour, and ends with:

```text
All document_processing tests passed.
```

### Use from Python or the notebook

```python
from document_processing import process_text, document_length, iter_collection

tokens = process_text("The Quick, Brown Fox!", use_stopwords=True, use_stemming=True)
print(tokens, document_length(tokens))

# Stream passages from the collection (after download / extract in the notebook)
for docid, pid, tokens, length in iter_collection("work/data/collection.tsv", max_lines=10):
    print(docid, pid, length, tokens[:12])
```

In the notebook, section **3.1 Document processing** imports the same module and applies `iter_collection` to `COLLECTION_TSV` with the `USE_STOPWORDS` / `USE_STEMMING` flags from the configuration cell.

