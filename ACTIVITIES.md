# Activities --- Set-up, C1, C2, C3, C4, C5, Notebook Structure, Report

---

## Prerequisite — Set-up and configuration cell

1. **Persist work across sessions**
Mount Google Drive in Colab (or set up equivalent local persistence).
Create the project folder on Drive for the final index, intermediate runs, statistics and artefacts.
At the start of each session, copy the index from Drive to local disk (`/content`) so queries are fast.
Keep the uncompressed collection only on local disk and re-download it when needed.

2. **Pinned installs**
`pip install package==x.y.z` for: `ir_datasets`, `ir_measures`, the stemmer (`PyStemmer` or `nltk.stem`, Porter / Snowball English), `scipy`, `numpy`, `pandas`, `tqdm`, `psutil`, and a plotting library.
Load a stop-word list and **cite its source**.
Allowed libraries only (PROJECT.md §3); no indexing / compression / ranking / DB / general-purpose-compressor libraries applied to the index.

3. **Runtime characteristics**
Print Python version, number of CPU cores, total RAM.

4. **Configuration cell (single cell, all paths and flags)**
   - Flags: `USE_STEMMING`, `USE_STOPWORDS` (each switchable so their effect can be measured).
   - BM25 parameters: `k1 = 0.9`, `b = 0.4`.
   - `BUFFER_SIZE` for the in-memory posting buffer (must be small enough to exercise the memory-bounded path; the indexer must work unchanged with a buffer **10× smaller**).
   - Paths: collection file, intermediate-runs directory, final index directory (lexicon, postings file(s), document table, statistics), run-file directory, Drive project folder.
   - Load the stop-word list (with source citation) and the stemmer.

---

## C1 — Document processing

1. **Single processing function**
`process_text(text: str) -> list[str]`, applied **identically** to passages and queries.
Stemming and stop-word removal toggled by the config flags.

2. **Case normalisation**
Lower-case the text.

3. **Punctuation / non-alphanumeric removal**
Replace every character that is not a letter or digit (`not c.isalnum()`) with a space (assignment default), or a justified variant documented in the report.

4. **Tokenisation**
Split on whitespace and drop empty strings.

5. **Stop-word removal (flag)**
Filter tokens present in the stop-word list when `USE_STOPWORDS` is on.

6. **Stemming (flag)**
Apply Porter / Snowball when `USE_STEMMING` is on; **cache the stem of each distinct token** for performance.

7. **Malformed input handling**
   - Decode every line as UTF-8 with `errors="replace"`; never fail on invalid bytes.
   - Skip lines without a tab or whose `pid` is not an integer; **count them**.
   - Keep passages whose text is empty after processing: they receive a valid `docid` and length 0.
   - Report the number of skipped lines and of empty passages.

8. **Document length**
`|d|` = number of tokens after processing, stop words excluded when `USE_STOPWORDS` is on.
Length-0 documents count in `N` and in `|d|_avg`.

---

## C2 — Inverted index

1. **On-disk layout (write the specification in the report)**
   - One (or a few) posting files; fixed-width binary `(docid, tf)` in **increasing docid order**; never one file per term.
   - Lexicon: `term -> {df, cf, offset, length}`.
   - Document table: `docid -> (pid, length)` — stored explicitly; `docid` assigned consecutively from 0 in the order passages are read (may or may not equal `pid`).
   - Statistics: `N` and `|d|_avg` (length-0 documents included).
   - Lexicon / doc table / stats: `pickle` or `numpy`. Postings: fixed-width binary in the core.

2. **Stream the collection**
Walk `collection.tsv` line by line, parse, apply `process_text`, emit `(docid, pid, tokens, length)` one passage at a time.
Never load the collection into memory.

3. **Bounded in-memory buffer**
Accumulate postings in memory; when `BUFFER_SIZE` is exceeded, sort and flush to disk as a **sorted intermediate run**.
**Restartable**: write each run to Drive as soon as it is complete and skip runs that already exist on re-run.

4. **Document table and statistics during the scan**
Store `pids[docid]` and `lengths[docid]` in `array` / `numpy` (not dicts). Count `N`, accumulate total length for `|d|_avg`; include length-0 documents.

5. **Merge intermediate runs**
Merge all runs into one postings file, writing each list with increasing docid and its corresponding `tf`;
build the lexicon with `df`, `cf`, `offset`, `length`.
**Report run-building time and merge time separately.**

6. **Persist the final index**
Write lexicon, postings file(s), document table and statistics to disk.
**Report the size on disk of each component** (postings, lexicon, document table).

7. **Query-time read API**
Load lexicon, document table and statistics into memory at start-up.
Read posting lists **from disk, only for the terms of the current query**, via ordinary reads or `mmap`.
Do **not** preload all postings and do **not** cache decoded posting lists across queries.

8. **Validate the memory-bounded path**
Index a subset (e.g. first 100k lines) with the normal buffer and with a buffer 10× smaller;
the two final indexes must be **identical**.

---

## C3 — Query processing

1. **DAAT search function**
`search(query: str, k: int, mode: str, scoring: str) -> list[tuple[int, float]]`, with `mode ∈ {"and","or"}` and `scoring ∈ {"tfidf","bm25"}`.
Return at most `k` `(pid, score)` pairs in decreasing score order. **Single-threaded** at query time.

2. **Query preprocessing**
Apply the same `process_text` as for passages; keep each distinct resulting term **once**;
a query with no terms left returns an empty result.

3. **Term resolution**
   - OR: ignore terms missing from the lexicon.
   - AND: if **any** term is missing from the lexicon, return an empty result.

4. **TF–IDF**
`(1 + log tf(t,d)) · log(N / df(t))`, summed over `t ∈ q ∩ d`.

5. **BM25**
`log(1 + (N − df + 0.5) / (df + 0.5)) · ((k1 + 1) · tf) / (tf + k1 · (1 − b + b · |d| / |d|_avg))`,
summed over `t ∈ q ∩ d`, `k1` and `b` from the configuration cell.

6. **DAAT OR and DAAT AND**
Document-at-a-time over the union / intersection of the posting lists; score candidates;
keep the best `k` in a **min-heap of size k**; **ties broken by increasing docid**.

7. **Latency measurement**
Time each `search` call with `time.perf_counter()`, including query processing and posting-list reads, excluding index load and run-file writes.
Warm up once over the whole query set (first 500 queries on dev-small) before timing.

---

## C4 — Evaluation

1. **Required runs, k = 1000**
   - **R1**: BM25, OR, index with stemming + stop-word removal → DL 2019, DL 2020, dev-small.
   - **R2**: TF-IDF, OR, same index → DL 2019, DL 2020, dev-small.
   - **R3**: BM25, AND, same index → DL 2019, DL 2020, dev-small.
   - **R4**: BM25, OR, index **without** stemming and **without** stop-word removal → DL 2019, DL 2020.

   Run-file format (TREC): `<qid> Q0 <pid> <rank> <score> <run-tag>` - ranks from 1, scores non-increasing, run tag a name without spaces identifying the configuration, **pids not docids**.
   A query with no results emits no lines.
   Store run files on Drive; do not submit them.

2. **Queries and qrels**
`ir_datasets.load("msmarco-passage/trec-dl-2019/judged")`, `"msmarco-passage/trec-dl-2020/judged"`, `"msmarco-passage/dev/small"`.
Pass queries through the same `process_text` as passages.
Use only the judged queries.

3. **Effectiveness with `ir_measures`**
   - DL 2019 / DL 2020: `nDCG@10`, `AP(rel=2)`, `R(rel=2)@1000`.
   - dev-small: `RR@10`, `R@1000`.
   Use `ir_measures.iter_calc` per query and **average yourself over all judged queries** - queries with no results (frequent in AND mode) count as 0.

4. **Efficiency**
   - Indexing time: wall-clock from reading the first line of the collection to writing the final index; report run-building and merge separately; exclude download.
   - Peak memory: `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (KB on Linux). Restart the runtime before indexing so earlier cells do not inflate it.
   - Index size: bytes on disk per component (postings, lexicon, document table).
   - Query latency: mean, median, p95 in ms, for every run of §7.1.

5. **Comparisons and significance**
   - TF-IDF vs BM25 (R2 vs R1).
   - AND vs OR (R3 vs R1), effectiveness and latency.
   - Stemming + stop-word effect (R4 vs R1), effectiveness, index size, latency.
   - One comparison per implemented extension.
   Two-sided paired t-test (`scipy.stats.ttest_rel`) on per-query `nDCG@10` for each TREC DL set and on `MRR@10` for dev-small; report p-values at α = 0.05.

6. **Sanity check**
Target R1: `nDCG@10 ≈ 0.50` (DL 2019), `≈ 0.48` (DL 2020), `MRR@10 ≈ 0.18` (dev-small).
Large deviations usually mean a bug - check query processing, IDF formula, document lengths, run files containing pids.

---

## C5 — Testing

1. **Toy collection**
A few hand-written documents (including punctuation, non-ASCII characters, an empty-after-processing passage, a malformed line) and a small query set.

2. **Document processing tests**
Hand-computed expected token lists for strings covering punctuation, non-ASCII and empty-after-processing passages;
malformed-line and skipped-line counting.

3. **Index construction tests**
Lexicon (`df`, `cf`, offsets, lengths), posting lists (increasing docids, correct tfs), document table and statistics must match values computed by hand.

4. **Merging test**
Build the index with a buffer so small that it forces several intermediate runs;
the final index must be **identical** (posting lists and lexicon) to the one built with a single run.

5. **Query processing tests**
For both modes (`and`, `or`) and both scoring functions (`tfidf`, `bm25`),
`search` must return the same results as a brute-force implementation that scores every document of the toy collection directly from the formulas.
Also test: empty queries, missing terms (OR ignores, AND empty), empty AND results, tie-breaking by increasing docid, top-k order.

6. **Harness**
Plain `assert` statements; runs in a few minutes; does not require the full collection.

---

## Notebook structure

Notebook must be organised in this order:
1. Set-up: pinned installs, Drive mount, configuration cell, runtime information.
2. Data: collection download, queries, qrels.
3. Implementation: document processing, indexing, query processing, scoring, extensions.
4. Tests (C5).
5. Indexing runs: time, peak memory, index size.
6. Evaluation and experiments: every table and plot of the report.
7. Report.

The notebook must run **top-to-bottom on a fresh free runtime**, after editing only the configuration cell.
Long steps save their output to Drive and are skipped when that output already exists.
Submit with outputs saved (short — do not print millions of lines).

---

## Report (PROJECT.md §8)

Sections (~10 pages equivalent):
1. What the program can do.
2. Internals: architecture, major functions and modules, data structures, file layout, design decisions.
3. Experimental set-up: runtime characteristics, data, protocol.
4. Results: indexing time, query latency, index size, effectiveness; the comparisons of §7.4 with significance.
5. Extensions implemented and their comparisons.
6. Limitations and what you would do differently.
7. Contribution of each team member; declared use of AI assistants.
