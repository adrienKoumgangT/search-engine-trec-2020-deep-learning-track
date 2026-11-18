from __future__ import annotations

import pickle
import tarfile
import time
from pathlib import Path

from tqdm import tqdm

from configuration import WORK, DATA_DIR, COLLECTION_TGZ, COLLECTION_TSV, PID_MAP_FILE, COLLECTION_TOKENS_PKL
from document_processing import iter_collection, process_text


def ensure_extracted() -> None:
    """Extract collection.tsv from the tarball if not already present."""
    if COLLECTION_TSV.exists():
        print(f"Already extracted: {COLLECTION_TSV}")
        return
    if not COLLECTION_TGZ.exists():
        raise FileNotFoundError(f"Missing tarball: {COLLECTION_TGZ}")

    print(f"Extracting {COLLECTION_TGZ.name}...")
    with tarfile.open(COLLECTION_TGZ, "r:gz") as tf:
        member = tf.getmember("collection.tsv")
        tf.extract(member, path=DATA_DIR)
    print(f"{COLLECTION_TSV} ({COLLECTION_TSV.stat().st_size / 1e9:.2f} GB)")


def build_token_cache(
    tsv_path: Path = COLLECTION_TSV,
    out_path: Path = COLLECTION_TOKENS_PKL,
    *,
    use_stopwords: bool = True,
    use_stemming: bool = True,
    max_lines: int | None = None,
) -> None:
    """
    Stream collection.tsv and write a pickle of:
        {
            "config": {"use_stopwords": ..., "use_stemming": ...},
            "passages": [(docid, pid, tokens, length), ...]
        }
    """
    if out_path.exists():
        print(f"Token cache already exists: {out_path} "
              f"({out_path.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Processing {tsv_path.name} "
          f"(stopwords={use_stopwords}, stemming={use_stemming})...")
    t0 = time.time()

    # We need an approximate total for tqdm. 8,841,823 is the MS MARCO v1
    # collection passage count; if you use max_lines, clamp it.
    total_passages = 8_841_823
    if max_lines is not None:
        total_passages = min(total_passages, max_lines)

    passages: list[tuple[int, int, list[str], int]] = []
    lengths_sum = 0
    lengths_min: int | None = None
    lengths_max = 0

    with tqdm(total=total_passages, unit="passage", desc="processing") as bar:
        for docid, pid, tokens, length in iter_collection(
            tsv_path,
            use_stopwords=use_stopwords,
            use_stemming=use_stemming,
            max_lines=max_lines,
            report=False,  # tqdm is the progress reporter here
        ):
            passages.append((docid, pid, tokens, length))
            lengths_sum += length
            lengths_min = length if lengths_min is None else min(lengths_min, length)
            lengths_max = max(lengths_max, length)
            bar.update(1)

    n = len(passages)
    elapsed = time.time() - t0
    avg_len = lengths_sum / n if n else 0.0

    print(f"Processed {n:,} passages in {elapsed:.1f}s "
          f"({n / elapsed:,.0f} passages/s)")
    print(f"Avg |d|: {avg_len:.2f}  min: {lengths_min}  max: {lengths_max}")

    print(f"Writing {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "config": {
                    "use_stopwords": use_stopwords,
                    "use_stemming": use_stemming,
                },
                "passages": passages,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_extracted()
    build_token_cache()

    # Quick load-back sanity check
    with open(COLLECTION_TOKENS_PKL, "rb") as f:
        cache = pickle.load(f)
    print(f"Config: {cache['config']}")
    print(f"Loaded {len(cache['passages']):,} passages")
    docid, pid, tokens, length = cache["passages"][0]
    print(f"Sample: docid={docid} pid={pid} |d|={length} tokens={tokens[:10]}...")
