# Retrieval ablation

27 answerable questions; a hit = a retrieved chunk contains the evidence string (whitespace-insensitive). k = 5.

| Config | hit@1 | hit@5 | MRR |
|---|---|---|---|
| A baseline (pypdf, plain chunks, no prefix, L2) | 11/27 | 20/27 | 0.54 |
| B + PyMuPDF heading-aware chunks (dense) | 18/27 | 23/27 | 0.723 |
| C + nomic task prefixes (dense) | 18/27 | 23/27 | 0.725 |
| D + hybrid BM25/RRF (shipped) | 17/27 | 25/27 | 0.76 |

## Rank of first evidence chunk per question (– = miss)

| Q | A | B | C | D |
|---|---|---|---|---|
| R1 | – | 1 | 1 | 1 |
| R2 | – | 1 | 1 | 5 |
| R3 | 1 | 1 | 1 | 1 |
| R4 | 2 | 1 | 1 | 1 |
| R5 | 2 | 1 | 1 | 1 |
| R6 | 1 | 5 | 4 | 2 |
| R7 | 1 | 1 | 1 | 1 |
| R8 | – | – | – | 2 |
| R9 | 1 | 1 | 1 | 1 |
| R10 | – | – | – | – |
| C1 | 1 | 4 | 1 | 1 |
| C2 | 2 | 1 | 1 | 1 |
| C3 | – | 1 | 1 | 1 |
| C4 | 1 | 1 | 1 | 1 |
| C5 | 3 | 4 | 3 | 2 |
| C6 | 5 | – | – | 2 |
| C7 | 1 | 1 | 1 | 1 |
| C8 | 2 | 1 | 3 | 3 |
| O1 | 5 | 3 | 3 | 2 |
| O2 | 1 | 1 | 1 | 1 |
| O3 | 3 | 1 | 1 | 1 |
| O4 | 2 | 1 | 1 | 1 |
| O5 | 1 | 1 | 1 | 1 |
| O6 | – | 1 | 1 | 1 |
| T1 | 1 | 2 | 3 | 2 |
| V1 | – | – | – | – |
| F1 | 1 | 1 | 1 | 1 |

## Relevance-gate scores

| Gate | answerable questions | unanswerable / off-topic |
|---|---|---|
| Baseline: mean 1/(1+L2) over top-5 (threshold 0.3) | min 0.489 · median 0.636 · max 0.691 | min 0.453 · median 0.572 · max 0.572 |
| Final: best cosine among hybrid top-5 (threshold 0.62) | min 0.547 · median 0.789 · max 0.889 | min 0.500 · median 0.660 · max 0.667 |
