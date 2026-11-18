# Refusal-threshold calibration

The gate score is the best cosine similarity between the **current question alone** and its hybrid top-5 chunks. Embeddings come from nomic-embed-text via Ollama 0.40.16 (Docker), with the `search_query:` prefix. Measured 2026-11-18 through `POST /api/retrieve`.

## Handbook questions (from `results.json`, final-llama8b)
- Lowest English handbook question: **R8** "Can I graduate with honors if I don't do the capstone?" scored **0.636**. The median is about 0.79.
- **V1** (the Vietnamese question) scored **0.546** and was refused. Retrieval also missed it in every configuration (A–D), because the English-only embedder doesn't match it to the English text.
- In-domain questions the handbook can't answer: U1 (tuition) 0.667 and U2 (head of department) 0.660. **No threshold can separate these** from answerable questions; the grounded prompt handles them, and llama3.1:8b declined both.

## Off-topic questions
| Score | Question | Refused at 0.62? |
|---|---|---|
| 0.824 | What is machine learning? *(arguably on-topic: CS307 describes ML)* | no (answered from CS307) |
| 0.728 | How many students attend Fulbright? *(university-related, not in the handbook)* | no; LLM must decline |
| 0.638 | Explain how transformers work in deep learning | no; llama: "I couldn't find this in the handbook." |
| 0.632 | Recommend a good laptop for programming | no; llama: "I couldn't find this in the handbook." |
| 0.615 | What is the weather in Ho Chi Minh City today? | yes |
| 0.612 | How do I apply for a student visa to the US? | yes |
| 0.601 | What is the time complexity of quicksort? | yes |
| 0.537 | Who is the president of the United States? | yes |
| 0.528 | How do I reverse a linked list in Python? | yes |
| 0.515 | What are the dining hall opening hours? | yes |
| 0.500 | What is the capital of France? | yes |
| 0.467 | Write me a poem about the ocean | yes |
| 0.458 | Tell me a joke | yes |

## Decision
Threshold **0.62**. It refuses 8 of the 10 clearly off-topic questions and refuses no English handbook question (the lowest is 0.636).

- Raising it would start refusing R8.
- Lowering it lets the weather and visa questions through.

The score distributions overlap, so the gate is a cheap first filter, not a guarantee. The grounded prompt's not-found rule is the second line of defense, and with llama3.1:8b it held for every leaked question we tried. qwen2.5:0.5b does not reliably follow that rule (see U2 in `answers.md`).

Caveat: this calibration uses a small sample. It also depends on the embedding runtime: the same questions scored about 0.1 higher for off-topic text under the host's Ollama 0.21, so re-calibrate if the Ollama version or the embedding model changes.

## Earlier bug found during calibration
The gate originally used the history-augmented retrieval query (previous question plus current question). An off-topic question asked right after an on-topic one inherited that relevance: World Cup 0.704 and sourdough 0.737 after a CS question, both of which would have passed. The gate now scores the current question alone; retrieval still uses the combined query so follow-ups work. Follow-ups scored alone came out at 0.61–0.75 ("And the minor?" 0.609 is the weakest; elliptical follow-ups are a known edge case).
