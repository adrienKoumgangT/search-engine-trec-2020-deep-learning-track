import ir_datasets
import numpy as np
import tarfile
import urllib.request
from tqdm.auto import tqdm
from pathlib import Path


# --- Configuration ---

WORK = Path.cwd() / 'work'
WORK.mkdir(exist_ok=True)

DATA_DIR = WORK / 'data'
DATA_DIR.mkdir(exist_ok=True)

COLLECTION_TGZ = DATA_DIR / 'collection.tar.gz'
COLLECTION_TSV = DATA_DIR / 'collection.tsv'
PID_MAP_FILE   = DATA_DIR / 'pid_to_docid.npy'

# --- Get Collection ---

COLLECTION_URL = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz'

def get_collection_and_extract(
        collection_tgz: Path = COLLECTION_TGZ,
        collection_tsv: Path = COLLECTION_TSV,
        collection_url: str = COLLECTION_URL,
        chunk_size: int = 1 << 20
):
    if collection_tgz.exists():
        print(f"Already present, skipping download: {collection_tgz}")
        return

    print("Downloading from msmarco.z22.web.core.windows.net…")

    # !wget - q - -show - progress - O "{collection_tgz}" "{collection_url}"

    with urllib.request.urlopen(collection_url) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with open(collection_tgz, "wb") as out, \
             tqdm(total=total, unit="B", unit_scale=True, desc="collection.tar.gz") as bar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))

    print(f"{collection_tgz.stat().st_size / 1e9:.2f} GB")


    if collection_tsv.exists():
        print(f"Already extracted : {collection_tsv}")
        return

    print(f"Extracting {COLLECTION_TGZ.name}…")
    with tarfile.open(COLLECTION_TGZ, 'r:gz') as tf:
        # The archive contains one file 'collection.tsv'
        member = tf.getmember('collection.tsv')
        tf.extract(member, path=DATA_DIR)
    print(f"{collection_tsv} ({collection_tsv.stat().st_size / 1e9:.2f} GB)")


# --- Robust parser with handling for "messy" inputs

def parse_collection(path, max_lines=None, verbose=True):
    """
    Parse collection.tsv handling the malformed lines.

    Returns
    -------
    pid_to_docid : dict[int, int]
    docid_to_pid : np.ndarray (int64), size = number of passages kept
    stats        : dict with 'kept', 'skipped', 'empty'
    """
    pid_to_docid = {}
    docid_to_pid = []
    skipped = 0
    empty = 0

    # text mode: universal newlines + replacement decoder
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        it = enumerate(f)
        if verbose:
            it = tqdm(it, total=max_lines or None, desc='Parsing', unit='line')

        for line_no, line in it:
            if max_lines is not None and line_no >= max_lines:
                break

            # remove only the line terminator(s)
            text_line = line.rstrip('\r\n')

            # split on the FIRST tab only : preserve the tabs in the text
            parts = text_line.split('\t', 1)
            if len(parts) != 2:
                skipped += 1
                continue

            pid_str, text = parts
            try:
                pid = int(pid_str)
            except ValueError:
                skipped += 1
                continue

            docid = len(docid_to_pid)          # consecutive from 0
            pid_to_docid[pid] = docid
            docid_to_pid.append(pid)

            if text == '':                     # passage empty "after processing"
                empty += 1

    docid_to_pid = np.asarray(docid_to_pid, dtype=np.int64)
    stats = {
        'kept':    len(docid_to_pid),
        'skipped': skipped,
        'empty':   empty,
    }
    return pid_to_docid, docid_to_pid, stats


def parsing_dev_subset(max_lines:int = 100_000):
    pid_to_docid_dev, docid_to_pid_dev, stats_dev = parse_collection(
        COLLECTION_TSV, max_lines=max_lines
    )

    print(f"Guarded passages : {stats_dev['kept']:,}")
    print(f"Skipped lines    : {stats_dev['skipped']:,}")
    print(f"Empty passages   : {stats_dev['empty']:,}")

    return pid_to_docid_dev, docid_to_pid_dev, stats_dev


def parsing_all(pid_map_file: Path = PID_MAP_FILE):
    if pid_map_file.exists():
        print(f"Loading existing mapping: {pid_map_file}")
        docid_to_pid = np.load(pid_map_file)
        pid_to_docid = {int(pid): docid for docid, pid in enumerate(docid_to_pid)}
        stats = {'kept': len(docid_to_pid)}
        print(f"{stats['kept']:,} passages")
    else:
        pid_to_docid, docid_to_pid, stats = parse_collection(COLLECTION_TSV)
        print(f"Guarded passages : {stats['kept']:,}")
        print(f"Skipped lines    : {stats['skipped']:,}")
        print(f"Empty passages   : {stats['empty']:,}")
        np.save(pid_map_file, docid_to_pid)
        print(f"Mapping saved : {pid_map_file}")

    return pid_to_docid, docid_to_pid, stats


def read_passage(docid_to_pid, docid):
    pid = int(docid_to_pid[docid])
    with open(COLLECTION_TSV, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            # simple scan ; in production, save an offset per docid
            if i == docid:
                parts = line.rstrip('\r\n').split('\t', 1)
                return pid, parts[1] if len(parts) == 2 else ''
    raise IndexError(docid)



def read_some_passage(docid_to_pid, first_docid: int = 0, last_docid: int = 10):
    if first_docid < 0 or last_docid < 0:
        raise IndexError(first_docid, last_docid)
    if last_docid < first_docid:
        last_docid = first_docid + 5
    for d in range(first_docid, last_docid + 1):
        pid, text = read_passage(docid_to_pid, d)
        print(f"docid={d}  pid={pid}  text={text[:80]!r}")


DATASETS = {
    'trec-dl-2019':      'msmarco-passage/trec-dl-2019/judged',
    'trec-dl-2020':      'msmarco-passage/trec-dl-2020/judged',
    'msmarco-dev-small': 'msmarco-passage/dev/small',
}

def load_queries_and_qrels(name, datasets=None):
    if datasets is None:
        datasets = DATASETS

    ds = ir_datasets.load(datasets[name])
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels = [
        (q.query_id, q.doc_id, int(q.relevance))
        for q in ds.qrels_iter()
    ]
    return queries, qrels


def load_qq(datasets=None):
    if datasets is None:
        datasets = DATASETS

    for name in datasets:
        queries, qrels = load_queries_and_qrels(name)
        graded_levels = {r for _, _, r in qrels}  # set, ex. {0, 1, 2, 3}
        print(f"{name:20s}  queries={len(queries):>5}  qrels={len(qrels):>7}  "
              f"graded_levels={sorted(graded_levels) if len(graded_levels) < 10 else 'n/a'}")




def qrels_to_docids(qrels, pid_to_docid):
    """Convert qrels (query_id, pid, rel) to (query_id, docid, rel).
    PIDs missing from the collection (dev subset) are ignored."""
    out = []
    missing = 0
    for qid, pid_str, rel in qrels:
        pid = int(pid_str)   # ir_datasets returns doc_id in str
        docid = pid_to_docid.get(pid)
        if docid is None:
            missing += 1
            continue
        out.append((qid, docid, rel))
    return out, missing


def test_qrels_to_docids():
    pid_to_docid, docid_to_pid, stats = parsing_all()

    queries_20, qrels_20 = load_queries_and_qrels('trec-dl-2020')
    qrels_20_docids, missing_20 = qrels_to_docids(qrels_20, pid_to_docid)
    print(f"convert qrels : {len(qrels_20_docids)}  (missing pids : {missing_20})")



if __name__ == '__main__':
    get_collection_and_extract()

    # to see some
    parsing_dev_subset()

    pid_to_docid, docid_to_pid, stats = parsing_all()

    read_some_passage(docid_to_pid, 0, 10)


    load_qq()

    test_qrels_to_docids()
