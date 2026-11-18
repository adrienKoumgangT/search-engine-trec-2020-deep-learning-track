from __future__ import annotations

import humanize
import numpy as np
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from itertools import islice

from tqdm import tqdm

from configuration import COLLECTION_TOKENS_PKL, LEXICON_PKL, INV_PKL, INDEX_PKL, STATS_PKL

"""
passages: list[tuple[int, int, list[str], int]] = [(docid, pid, tokens, length)]

{
    "config": {
        "use_stopwords": use_stopwords,
        "use_stemming": use_stemming,
    },
    "passages": passages,
},
"""


def ensure_preprocessed() -> None:
    if not COLLECTION_TOKENS_PKL.exists():
        raise FileNotFoundError(f"Missing collections tokens: {COLLECTION_TOKENS_PKL}")

@dataclass
class LexiconEntry:
    termid: int
    df: int = 0
    cf: int = 0


def build_index(passages, batch_size=100):
    lexicon = {}
    doc_index = []
    inv_d, inv_f = {}, {}
    termid = 0

    num_docs = 0
    total_dl = 0

    vocab_lengths = [0]              # |V| over time
    collection_lengths = [0]          # N (total tokens) over time
    batch_token_buffer = []          # accumulate tokens for batching

    def _vocab_size():
        return len(lexicon)

    def _collection_size():
        return sum(entry.cf for entry in lexicon.values())

    for passage in tqdm(passages, desc='Indexing', total=len(passages)):
        docid, pid, tokens, length = passage

        # compute term frequencies in the document
        token_tf = Counter(tokens)

        for token, tf in token_tf.items():
            # add new term to lexicon and to inverted index
            if token not in lexicon:
                lexicon[token] = LexiconEntry(termid)  # assign a new termid
                inv_d[termid], inv_f[termid] = [], []  # initialize posting list
                termid += 1
            token_id = lexicon[token].termid

            # append docid in the posting list of the term
            inv_d[token_id].append(num_docs)
            inv_f[token_id].append(tf)  # store also the term frequency

            # update lexicon statistics
            lexicon[token].df += 1
            lexicon[token].cf += tf

        doclen = len(tokens)

        # store the document length in the document index
        doc_index.append((docid, doclen))

        # update global statistics
        total_dl += doclen
        num_docs += 1

        # Update Heaps' law style statistics in batches
        batch_token_buffer.extend(tokens)
        while len(batch_token_buffer) >= batch_size:
            # consume a batch worth of tokens
            del batch_token_buffer[:batch_size]
            vocab_lengths.append(_vocab_size())
            collection_lengths.append(_collection_size())

    # Flush any remaining tokens in the buffer
    if batch_token_buffer:
        del batch_token_buffer[:]
        vocab_lengths.append(_vocab_size())
        collection_lengths.append(_collection_size())
    # --- End statistics tracking ---

    # Sorted frequency list
    sorted_freqs = sorted(
        (entry.cf for entry in lexicon.values()), reverse=True
    )

    # compute some collection statistics
    stats = {
        'num_docs': num_docs,          # collection size
        'num_terms': len(lexicon),      # vocabulary size
        'num_tokens': total_dl,

        'vocab_lengths': vocab_lengths,
        'collection_lengths': collection_lengths,
        'sorted_freqs': sorted_freqs,   # term frequencies, descending
    }
    return lexicon, {'docids': inv_d, 'freqs': inv_f}, doc_index, stats


if __name__ == '__main__':
    ensure_preprocessed()

    if not LEXICON_PKL.exists():

        # Quick load-back sanity check
        with open(COLLECTION_TOKENS_PKL, "rb") as f:
            collection_tokens = pickle.load(f)

        passages_collection = collection_tokens["passages"]
        print(f"Passages collection: {len(passages_collection)}")
        print(f"Some passages:")
        for passage in passages_collection[:5]:
            docid, pid, tokens, length = passage
            print(f"Sample: docid={docid} pid={pid} |d|={length} tokens={tokens[:10]}...")

        print()
        print(f"Indexing {len(passages_collection)} passages")
        lex, inv, doc, stats = build_index(passages_collection)

        with open(LEXICON_PKL, 'wb') as f:
            pickle.dump(lex, f)

            print()
            print(f'The lexicon requires {humanize.naturalsize(os.path.getsize(LEXICON_PKL))}')

        with open(INV_PKL, 'wb') as f:
            pickle.dump(inv, f)

            print()
            print(f'The inverted requires {humanize.naturalsize(os.path.getsize(INV_PKL))}')

        with open(INDEX_PKL, 'wb') as f:
            pickle.dump(doc, f)

            print()
            print(f'The index requires {humanize.naturalsize(os.path.getsize(INDEX_PKL))}')

        with open(STATS_PKL, 'wb') as f:
            pickle.dump(stats, f)

            print()
            print(f'The stats requires {humanize.naturalsize(os.path.getsize(STATS_PKL))}')

    else:
        print(f"Loading index from {LEXICON_PKL}")
        with open(LEXICON_PKL, "rb") as f:
            lex = pickle.load(f)

        print()
        print(f"Loading index from {INV_PKL}")
        with open(INV_PKL, "rb") as f:
            inv = pickle.load(f)

        print()
        print(f"Loading index from {INDEX_PKL}")
        with open(INDEX_PKL, "rb") as f:
            doc = pickle.load(f)

        print()
        print(f"Loading stats from {STATS_PKL}")
        with open(STATS_PKL, "rb") as f:
            stats = pickle.load(f)


    print()
    for l in list(islice(lex.items(), 10)):
        print(l)

    print()
    print(f'termid 0 -> {[(d, f) for d, f in zip(inv['docids'][0], inv['freqs'][0])]}')

