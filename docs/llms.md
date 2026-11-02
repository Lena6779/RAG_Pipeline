# Running LLMs Locally with Ollama

Ollama is a free tool for running large language models on your own computer. Once installed, it runs a local server at http://localhost:11434 that applications can send requests to.

Models are downloaded with the command "ollama pull" followed by the model name. For example, "ollama pull llama3.2:1b" downloads a small 1-billion-parameter version of Meta's Llama 3.2 model, which runs well on most laptops.

Smaller models are faster and use less memory, but they follow instructions less reliably. A 1B model may sometimes ignore rules like citing sources, while larger models such as 8B versions are more consistent but slower.

Ollama's chat endpoint is /api/chat. It accepts a model name, a list of messages with roles like system and user, and a stream setting. When streaming is on, Ollama sends the answer one token at a time as separate lines of JSON.

Running models locally keeps data private because nothing is sent to an outside company. It also has no per-request cost, which makes it good for learning and experimenting.