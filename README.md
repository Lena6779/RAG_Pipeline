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
  "answer": "[Answer from the model] (llms.md)",
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

The test run started with a `DISTANCE_THRESHOLD` of 1.0.

| Query type | Question | Best distance | Confidence | Chunks retrieved | Result |
|---|---|---|---|---|---|
| In-scope | What command downloads a model in Ollama? | [fill in] | [fill in] | [fill in] | [fill in] |
| Partially in-scope | What command downloads a model in Ollama, and how much does a GPU cost? | [fill in] | [fill in] | [fill in] | [fill in] |
| Out-of-scope | What is the capital of France? | [fill in] | [fill in] | [fill in] | [fill in] |
| Ambiguous | How does it work? | [fill in] | [fill in] | [fill in] | [fill in] |

### Threshold Tuning

In the baseline tests below, all three questions had a best distance under 1.0, so a threshold of 1.0 would not have rejected any of them.

After testing, I set `DISTANCE_THRESHOLD` to [final value] because [reason based on your results].

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
- Confidence is based on the best distance before filtering, so a rejected query can still show `medium` if the threshold is below 1.0.

## Possible Improvements

- **Larger model:** A model like `llama3.2:3b` follows instructions more reliably, at the cost of speed and memory.FF