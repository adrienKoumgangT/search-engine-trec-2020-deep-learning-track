from pathlib import Path


# --- Configuration ---

WORK = Path.cwd() / 'work'
WORK.mkdir(exist_ok=True)

DATA_DIR = WORK / 'data'
DATA_DIR.mkdir(exist_ok=True)

COLLECTION_TGZ = DATA_DIR / 'collection.tar.gz'
COLLECTION_TSV = DATA_DIR / 'collection.tsv'
PID_MAP_FILE   = DATA_DIR / 'pid_to_docid.npy'

OUT_PKL = DATA_DIR / "collection_tokens.pkl"
IN_COLAB = False
