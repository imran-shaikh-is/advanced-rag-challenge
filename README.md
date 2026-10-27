# Advanced RAG Challenge

Question answering over three PDF reports. Each report is structured differently, so each one gets its own ingestion and retrieval strategy. All three share one embedding model, one vector store, one answer generator and one evaluation harness.

| Document | Pages | What makes it hard | Approach |
|---|---|---|---|
| **GPO Annual Report** (FY 2018, US Government Publishing Office) | 82 | Long narrative, then audited financial statements | Page-aware text RAG |
| **World Bank Global Economic Prospects** (June 2026) | 200 | Chapters, boxes and regional sections; dense forecast tables | Hierarchical (section-aware) RAG with table-row records |
| **Apple iMac Product Environmental Report** | 12 | Borderless tables, a stacked bar chart, facts qualified by endnotes | Table-, chart- and footnote-aware RAG |

## Results

Each answerable question has a verified page and keywords. Every test set has 14 questions: 12 answerable and 2 whose answer is not in the report.

| Document | Page hit @5 | MRR | Keyword recall | Unanswerable handled | Avg latency |
|---|---|---|---|---|---|
| GPO | **1.00** | 0.92 | **0.97** | 2 / 2 | 5.4 s |
| World Bank | **1.00** | 0.76 | **0.93** | 2 / 2 | 11.9 s |
| Apple | **1.00** | 1.00 | **0.92** | 2 / 2 | 13.3 s |

- **Page hit @5:** a top-5 result comes from a page where the answer is stated.
- **MRR:** mean reciprocal rank of the first such result.
- **Keyword recall:** share of the expected facts that appear in the generated answer.
- **Unanswerable handled:** the model said the answer was not in the report, rather than inventing one. It said "not found" on no answerable question.
- **Avg latency:** retrieval, re-ranking and the Groq call, with `top_k=5` and `openai/gpt-oss-120b`.
- results are with gpt-oss-120b, set GROQ_MODEL to reproduce”

All figures come from one full run: `evaluation/results/eval_all_full_20261021_132828.json` (CSV alongside).

All 36 answerable questions were also read by hand, and every answer is correct. The keyword misses are phrasing differences or details the answer left out. For example, the model writes "four‑port model" where the keyword is "four ports", or omits that the audit was GPO's "22nd" consecutive clean opinion.

### Progress during the challenge

The first evaluation run exposed the weak spots. Each row below was re-measured after the change:

| Document | Before | After | Main change |
|---|---|---|---|
| World Bank | page hit 0.67 (section range), MRR 0.38, 32.6 s per question, 1 wrong number | page hit 0.92 (exact page), MRR 0.70, 11.0 s | Real section detection, table rows with column headers, exact-page child chunks, compact context window |
| Apple | page hit 0.92, MRR 0.83, Zero Waste answer invented "100%" | page hit 1.00, MRR 0.92, all answers correct | Endnote records and linking, rebuilt tables, chart description |
| All three | MRR 0.77 / 0.70 / 0.92; short questions such as "What is the global growth forecast for 2026?" answered "not found" | page hit 1.00 on all three; MRR 0.92 / 0.76 / 1.00 | Cross-encoder re-ranking of 30 vector candidates |

Two alternatives were measured on the World Bank set and dropped: hybrid BM25 + vector search (no gain over re-ranking alone, slower) and the `BAAI/bge-small-en-v1.5` embedding model (page hit 0.87 versus 0.93 for MiniLM).

## Architecture

```
                    ┌──────────────────────── Streamlit UI (app.py) ────────────────────────┐
                    │        choose document · ask question · answer · sources · evidence    │
                    └───────────────┬──────────────────────┬──────────────────────┬─────────┘
                                    │                      │                      │
                         src/pipelines/gpo_rag   world_bank_rag              apple_rag
                                    │                      │                      │
   ingestion        text_loader: pages →      world_bank_loader:          apple_loader:
                    700-char chunks           bookmarks + font headings   text chunks, table rows,
                                              → sections; table rows      chart, endnotes
                                    │                      │                      │
   retrieval        vector_store: chunks      hierarchical_retriever:     records,
                                              child search → window or    de-duplicated
                                              parent expansion
                                    └──────────────────────┼──────────────────────┘
                                                           │
                          ChromaDB (chroma_db/) · all-MiniLM-L6-v2 embeddings · cosine
                          → 30 candidates → reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 → top-k
                                                           │
                          src/generation/answer_generator: Groq LLM, temperature 0,
                          answers only from the retrieved context, cites source and page
```

### Retrieval: vector search + re-ranking (all documents)
The bi-encoder embeds the question and each passage separately. That works for long, specific questions but misses short ones. In the app, "What is the global growth forecast for 2026?" ranked regional outlooks above the pages that state 2.5 percent, and the model answered "not found".

Each retriever therefore takes 30 vector candidates and re-orders them with a cross-encoder (`src/retrieval/reranker.py`), which reads the question and passage together. The document-specific rules (World Bank section caps, Apple de-duplication) are then applied to the re-ranked list.

### GPO: page-aware text RAG
- `text_loader.py` extracts each page separately with PyMuPDF. It splits each page into 700-character chunks with 120 characters of overlap, so every chunk keeps the page it came from.
- The re-ranked top-k chunks go to the LLM. The sources list the distinct pages.

### World Bank: hierarchical RAG
The first version split sections on every all-caps or number-led line. That gave 470 sections, many of them titled with the running page header ("GLOBAL ECONOMIC PROSPECTS | JUNE 2026"). The rework in `world_bank_loader.py` uses the document's own structure:

- **Chapters** come from the PDF bookmarks: chapters, regional outlooks and the Statistical Annex. Boxes are bookmarked where they start but not where they end, so the loader detects box titles from the text instead.
- **Sections** are in-page subheadings, detected by font: Helvetica at 10.5pt or larger, plus 10pt "BOX x.y" titles. Body text is Garamond. Wrapped headings are re-joined, and running headers and "(continued)" box headers are skipped. The result is 116 sections such as *East Asia and Pacific › Outlook*.
- **Forecast tables:** plain text extraction flattens a row to "Euro area 0.5 1.0 1.4 0.8 …" and loses the year headers. The loader instead rebuilds each row from word positions, mapping every number to the year column above it. The result is 277 self-describing records, for example:
  `Romania: 2023=2.3, 2024=0.9, 2025e=0.7, 2026f=0.0 … change from January 2026 projections: 2026f=-1.3`
- **Child chunks** are 600 characters long, split page by page, so each child keeps its exact page. Each child is prefixed with `[chapter > section]` so the embedding knows where the passage sits.
- **Retrieval** searches the children, re-ranks them and then expands each hit:
  - `context="window"` (the default) returns the matched child with its neighbours from the same section. This cut latency from 32.6 s to about 11 s per question.
  - `context="parent"` returns the whole section.

  At most 2 hits come from any one section or table, so a single forecast table cannot fill every slot.
- **Text cleanup:** the report's Garamond font maps its "Th" ligature to a plain "T", so about 300 words came out as "Te", "Tis" and so on. The loader detects the ligature glyph by its width (1.07× the font size, against 0.66× for a real "T") and restores "Th". It also expands ligatures such as "ﬁ".

### Apple: tables, chart and endnotes
`apple_loader.py` turns the 12-page report into 57 small records. Each record keeps two fields: `content`, which is embedded, and `text`, which is content plus linked endnotes and is what the LLM reads.

- **Text:** paragraph chunks prefixed with the page title.
- **Endnotes:** footnote markers are superscripts, meaning digit spans set below 75% of their line's font size. Every endnote becomes its own record (19 in total). Every chunk that cites a marker also carries that endnote's text. For example, the Zero Waste claim on page 5 is linked to endnote 16, which names UL 2799 and the 90% threshold.
- **Tables:** these tables have no borders. PyMuPDF's detector finds only the shaded header bands, and on page 12 just 2 of 5 rows. The loader takes the columns from the header band and rebuilds the rows below it from word positions, stopping at the first vertical gap larger than a row gap. Bullet rows inherit their parent label (*Life cycle emissions › Production = 50%*). Each row is embedded on its own, so the right row is found, but the LLM receives the whole table. Otherwise "Production 50%" would arrive without "Product use 45%".
- **Chart:** the stacked bar chart is found from its drawings: at least 2 columns of 3 or more filled segments whose heights vary, which excludes table shading. Its labels are grouped into bars by x-position and ordered top to bottom. A leading summary line gives the bar totals: `445 kg Baseline emissions -> –99 kg Emissions reductions -> 346 kg iMac emissions`.
- **Retrieval** over-fetches, drops duplicate documents (the rows of one table share the same LLM text) and re-ranks.

## Setup

Requires Python 3.10+ and a [Groq](https://console.groq.com) API key.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b      # optional; default is llama-3.3-70b-versatile
```

The three reports are included under `data/`. They come from their publishers:

| File | Report | Source |
|---|---|---|
| `data/gpo/GPO-Anual-report.pdf` | U.S. Government Publishing Office, FY 2018 Annual Report | [gpo.gov](https://www.gpo.gov/docs/default-source/news-content-pdf-files/2018_annualreport.pdf) |
| `data/world_bank/GEP-Jun-2026.pdf` | World Bank, Global Economic Prospects, June 2026 | [World Bank Open Knowledge Repository](https://openknowledge.worldbank.org/entities/publication/e96798d7-a81d-44ea-8287-1b4d6ea83f75) |
| `data/apple_imac/apple_imac_report.pdf` | Apple, iMac Product Environmental Report, October 2024 | [apple.com](https://www.apple.com/environment/pdf/products/desktops/iMac_PER_Oct2024.pdf) |

The reports remain the property of their publishers and are included only so the project runs out of the box.

Build the indexes. The first run downloads the embedding and re-ranking models from Hugging Face. This also regenerates `data/apple_imac/extracted_records.json`:

```bash
python ingest.py                  # all three
python ingest.py world_bank       # or just one
```

Run the app:

```bash
streamlit run app.py
```

## Evaluation

```bash
python -m evaluation.run_eval                         # all documents, retrieval + answers
python -m evaluation.run_eval --doc apple             # one document
python -m evaluation.run_eval --retrieval-only        # no LLM calls, retrieval metrics only
python -m evaluation.run_eval --rescore evaluation/results/<run>.json   # re-score saved answers
```

- **Test sets:** `evaluation/<doc>_questions.json`. Each question has `expected_answer`, `expected_keywords`, `expected_pages` and a `category` (text, table, chart, footnote or unanswerable). Every keyword was checked programmatically against the text of its expected page.
- **Page numbers** are 1-based PDF page indexes, not the numbers printed on the pages. `python -m evaluation.find_page <doc> "<phrase>"` lists the pages that contain a phrase, for writing new questions.
- **Keyword matching** normalises Unicode spacing, dash variants, "percent"/"%" and hyphens between words, because the LLM often writes `91.9 %` or `four‑port`.
- **Rescore** re-reads the current question files, so a corrected test set can be applied to saved answers without spending API calls.
- **Output:** each run writes a JSON file (summary, every answer, retrieved evidence) and a CSV to `evaluation/results/`.

## Project layout

```
app.py                         Streamlit UI
ingest.py                      Builds all indexes
src/ingestion/
  text_loader.py               GPO: page-wise extraction and chunking
  world_bank_loader.py         World Bank: sections from bookmarks/fonts, table rows
  apple_loader.py              Apple: text, endnotes, tables, chart
src/retrieval/
  vector_store.py              GPO collection
  reranker.py                  Cross-encoder re-ranking shared by all retrievers
  hierarchical_retriever.py    World Bank parent/child collections and expansion
src/pipelines/                 One ingest_* / ask_* pair per document
src/generation/
  answer_generator.py          Groq call with grounded-answer prompt
evaluation/
  run_eval.py                  Evaluation runner and metrics
  find_page.py                 Phrase → page lookup for building test sets
  *_questions.json             Verified test sets
  results/                     Evaluation runs
```

## Known limitations

- **Chart segment sizes:** the Apple chart's bar totals and segment names come from the text layer. The size of each segment (how many kg are electricity versus materials) exists only as drawn shapes, so reading it would need a vision model.
- **Layout-specific heuristics:** the World Bank and Apple loaders rely on these reports' fonts, superscript sizes and table styling. A different report would need its thresholds re-tuned.
- **Small test sets:** 12 answerable questions per document is enough to find failures, not to estimate accuracy precisely. Keyword recall is a proxy for correctness, so the answers were also read by hand.
- **Rate limits:** on Groq's free tier, a full run of all 42 questions uses roughly 80–100k tokens, about half the daily allowance for `openai/gpt-oss-120b`. Use `--retrieval-only` while iterating.
- **GPO** uses plain page chunking, which scored well on this test set. Its financial statement tables are not parsed into rows the way the World Bank and Apple tables are.
