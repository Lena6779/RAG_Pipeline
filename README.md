# RAG Pipeline with ChromaDB and Ollama

A complete local Retrieval-Augmented Generation (RAG) pipeline with guardrails. It loads a folder of documents, stores them in a persistent ChromaDB vector database, retrieves the most relevant chunks for a question, filters out weak matches, and uses a local LLM through Ollama to generate an answer grounded in the remaining chunks. Each response includes the answer, its sources, a confidence level, and the number of chunks used.

Everything runs locally. No API keys or internet connection are needed after setup.

## How It Works

```
docs/ → chunk by paragraph → ChromaDB (embeddings)
                                    ↓
question → retrieve top 3 chunks → filter by distance
                                    ├─ no chunks pass → "no relevant information" response
                                    └─ chunks pass → build prompt → Ollama → structured response
```

1. **Ingest:** Loads every `.txt` and `.md` file in `docs/`, splits each file into paragraphs (skipping markdown headings), and stores the chunks in a persistent ChromaDB collection. ChromaDB creates the embeddings automatically using its default model (all-MiniLM-L6-v2).
2. **Retrieve:** Embeds the user's question and finds the 3 closest chunks using cosine distance.
3. **Filter:** Uses the best distance to set a confidence level, then keeps only chunks below `DISTANCE_THRESHOLD`. If no chunks pass, the pipeline returns a "no relevant information" response without calling the model.
4. **Generate:** Builds a prompt containing system rules plus the remaining chunks (each labeled with its source file), then sends it to `llama3.2:1b` through Ollama's `/api/chat` endpoint with a temperature of 0.
5. **Respond:** Returns a dictionary containing the answer, sources, confidence level, and number of chunks used.

### About Cosine Distance

The collection uses cosine distance to measure how similar a chunk is to the question. Cosine distance compares the *direction* of two embeddings rather than their exact values, so it captures similarity in meaning even when the wording is different.

- **Close to 0:** very similar meaning
- **Around 0.5:** somewhat related
- **Close to 1:** unrelated

The distances in the baseline test results below follow this scale: the answerable question's best match was 0.23, while the out-of-scope question's best match was 0.93.

## Features

- Paragraph-based chunking with source metadata
- Persistent vector database (`rag_db/`), so documents are only embedded once
- Four guardrails: distance filtering, confidence levels, a strengthened system prompt, and structured responses
- Source lists built in code from the chunks sent to the model
- Retrieved chunks (file, distance, preview, and pass/filtered status) are displayed before the response
- Graceful error messages if Ollama isn't running or the model isn't downloaded
- Automatic four-query test run on startup
- Interactive loop that runs until the user types `quit`

## Guardrails

| Guardrail | What it does |
|---|---|
| Distance filtering | Sends only chunks below `DISTANCE_THRESHOLD` to the model. If no chunks pass, it returns a "no relevant information" response without calling the model. |
| Confidence levels | Uses the best retrieved distance to return `high` (< 0.5), `medium` (< 1.0), or `low` (>= 1.0). |
| Strengthened system prompt | Tells the model not to make up information, to say "I don't know" when unsure, and to always cite its sources. |
| Structured response | Returns a dictionary with `answer`, `sources`, `confidence`, and `chunks_retrieved`. |

### Example Response

```json
{
  "answer": "The command to download a model in Ollama is \"ollama pull\". For example, \"ollama pull llama3.2:1b\" downloads a small 1-billion-parameter version of Meta's Llama 3.2 model.",
  "sources": ["llms.md"],
  "confidence": "high",
  "chunks_retrieved": 3
}
```

`chunks_retrieved` is the number of chunks that passed the distance filter and were sent to the model.

## Project Structure

```
RAG_Pipeline/
├── docs/
│   ├── embeddings.md
│   ├── llms.md
│   ├── prompt_engineering.md
│   ├── rag.md
│   ├── streamlit.md
│   └── vector_databases.md
├── .gitignore       # Keeps venv/ and rag_db/ out of GitHub
├── my_rag.py        # Terminal RAG pipeline with guardrails
└── README.md
```

## Setup

1. Install [Ollama](https://ollama.com) and download the model:
   ```
   ollama pull llama3.2:1b
   ```
2. Create and activate a virtual environment, then install the packages:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install chromadb requests
   ```
3. Make sure Ollama is running (open the Ollama app, or run `ollama serve`).

## Usage

```
python my_rag.py
```

On the first run, the documents are ingested (30 chunks from 6 files). Later runs reuse the saved database. To re-ingest after changing the documents, delete the `rag_db/` folder.

The program runs the four guardrail tests first, printing a structured response for each. It then starts the interactive prompt. Type a question at the `You:` prompt, or `quit` to exit.

To change how strict the filter is, edit `DISTANCE_THRESHOLD` at the top of `my_rag.py`.

## Guardrail Test Results

I ran all four query types at two thresholds: the assignment's starting value of 1.0, then 0.6. Both runs used a temperature of 0.

### Threshold 1.0

| Query type | Question | Best distance | Confidence | Chunks retrieved | Result |
|---|---|---|---|---|---|
| In-scope | What command downloads a model in Ollama? | 0.23 | high | 3 | Passed. Correctly answered `ollama pull`. |
| Partially in-scope | What command downloads a model in Ollama, and how much does a GPU cost? | 0.39 | high | 3 | Passed. Gave `ollama pull` and said the context did not cover GPU cost, although it opened with "I don't know." |
| Out-of-scope | What is the capital of France? | 0.93 | medium | 3 | Partly passed. The model said "I don't know," but the filter let unrelated chunks through and `sources` still listed files. |
| Ambiguous | How does it work? | 0.80 | medium | 3 | Partly passed. The model said "I don't know," but the filter let unrelated chunks through and `sources` still listed files. |

### Threshold 0.6

| Query type | Question | Best distance | Confidence | Chunks retrieved | Result |
|---|---|---|---|---|---|
| In-scope | What command downloads a model in Ollama? | 0.23 | high | 3 | Passed. Correctly answered `ollama pull`. |
| Partially in-scope | What command downloads a model in Ollama, and how much does a GPU cost? | 0.39 | high | 2 | Partly passed. It did not make up a GPU cost, but it wrongly said the download command was not specified while also quoting `ollama pull`. |
| Out-of-scope | What is the capital of France? | 0.93 | medium | 0 | Passed. All chunks were filtered, the model was not called, and `sources` was empty. |
| Ambiguous | How does it work? | 0.80 | medium | 0 | Passed. All chunks were filtered, the model was not called, and `sources` was empty. |

### Threshold Tuning

At 1.0, all 12 retrieved chunks passed the filter, so it did not block anything. The model refused the out-of-scope and ambiguous questions on its own, but that relied entirely on the model following its instructions.

I set `DISTANCE_THRESHOLD` to 0.6 because it falls between the in-scope best distance (0.23) and the out-of-scope and ambiguous best distances (0.93 and 0.80). At 0.6, those two questions are rejected in code before the model is called. The baseline "What is an LLM?" question had a best distance of 0.77, so it would also be filtered at this threshold, although it was not part of these test runs.

There was one tradeoff: the partially in-scope answer got worse at 0.6. It lost one chunk, and the model contradicted itself about the download command. This shows that the threshold can block unrelated questions, but it can't guarantee that a small model will handle a mixed question well.

## Baseline Test Results (Before Guardrails)

These tests show how the pipeline behaved before the guardrails were added.

### Test 1: Answerable from the documents

**Question:** What command downloads a model in Ollama?

```
Retrieved 3 chunks:
  [1] llms.md (distance: 0.2273)
  [2] llms.md (distance: 0.4376)
  [3] llms.md (distance: 0.5288)

Answer:
A model is downloaded with the command "ollama pull" followed by the model name.
For example, "ollama pull llama3.2:1b" downloads a small 1-billion-parameter
version of Meta's Llama 3.2 model.
```

**Result: Passed.** All 3 chunks came from the correct file, and the closest match (0.23) contained the exact answer. The answer was correct and used only information from the context. However, the model did not include a source citation despite the instruction.

### Test 2: Related but not directly in the documents

**Question:** What is an LLM?

```
Retrieved 3 chunks:
  [1] llms.md (distance: 0.7657)
  [2] rag.md (distance: 0.7959)
  [3] embeddings.md (distance: 0.8454)

Answer:
According to the [Source: llms.md](https://llms.md), an Llama is Meta's language
model. It's a type of transformer-based language model that is used for various
natural language processing tasks...
```

**Result: Failed.** None of the documents define an LLM, so the correct response was a refusal. Retrieval returned only weak matches (0.77–0.85). Instead of refusing, the model:
- Confused "LLM" with "Llama," a model mentioned in the retrieved chunk
- Added facts not in the context, such as "transformer-based," from its own training
- Turned the source label into a fake web link

This question also gave inconsistent results across runs. In earlier testing, it produced a made-up answer one time and a correct refusal another time.

### Test 3: Completely outside the scope

**Question:** What is the capital of France?

```
Retrieved 3 chunks:
  [1] prompt_engineering.md (distance: 0.9298)
  [2] streamlit.md (distance: 0.9536)
  [3] streamlit.md (distance: 0.9580)

Answer:
I don't have enough information to answer that based on the available documents.
(prompt_engineering.md)
```

**Result: Passed.** The chunks were unrelated (0.93–0.96), and the model correctly refused. It unnecessarily cited `prompt_engineering.md`, applying the citation rule even though no source was used. A second out-of-scope question ("What is the capital of the USA?") produced the same correct refusal.

### Test 4: Ollama not running

With Ollama stopped, the same question was asked:

```
Retrieved 3 chunks:
  [1] prompt_engineering.md (distance: 0.9298)
  ...

Answer:
ERROR: Cannot connect to Ollama. Is it running? (ollama serve)
```

**Result: Passed.** Retrieval still worked because ChromaDB handles embeddings on its own. When generation tried to reach Ollama, the connection error was caught and a helpful message was shown. The program kept running, so after restarting Ollama the next question worked without restarting the script.

## Baseline Analysis

| Test | Best distance | Result |
|---|---|---|
| Answerable | 0.23 | Correct answer |
| Related, not in docs | 0.77 | Made-up answer |
| Out of scope | 0.93 | Correct refusal |

The pipeline handled the two clear-cut cases well. When a chunk closely matched the question, the answer was accurate, and when nothing was related, the model refused.

The hardest case was the middle one, where the retrieved chunks were loosely related to the question. ChromaDB always returns the closest chunks even when none contain the answer, and the small 1B model treated that loosely related text as enough to build an answer from, mixing in its own training knowledge.

The model also followed the citation rule inconsistently: it skipped citations on a correct answer, invented a link on a wrong answer, and cited a file on a refusal.

These results led to the distance filter and the source list built in code.

## Limitations

- Distance measures similarity, not whether a chunk actually answers the question.
- The system prompt reduces unsupported answers but can't guarantee the model will follow every rule.
- Confidence is based on the best distance before filtering, so a rejected query can still show `medium` when the threshold is below 1.0. This happened in the out-of-scope and ambiguous tests at 0.6.
- Confidence reflects the closest chunk, not whether the whole question was answered. The partially in-scope question showed `high` even though the documents do not cover GPU cost.
- The model did not name a source in any of the four test answers, despite the citation rule. The `sources` field is built in code, so it still shows which files were sent to the model.

## Possible Improvements

- **Larger model:** A model like `llama3.2:3b` follows instructions more reliably, at the cost of speed and memory.