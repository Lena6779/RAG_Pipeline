# Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation, or RAG, is a technique that gives a language model access to outside information at the moment it answers a question. Instead of relying only on what the model learned during training, the system first searches a collection of documents and passes the most relevant pieces to the model as context.

A RAG pipeline has three main stages: retrieval, augmentation, and generation. Retrieval finds the document chunks most similar to the user's question. Augmentation inserts those chunks into the prompt. Generation is when the language model reads the prompt and writes an answer.

RAG reduces hallucinations because the model is told to answer only from the provided context. If the context does not contain the answer, a well-designed system prompt instructs the model to say it does not know rather than inventing something.

RAG is cheaper and faster to update than fine-tuning. To teach a RAG system new information, you simply add new documents to the database. Fine-tuning would require retraining the model itself, which takes much more time and computing power.

A common failure in RAG systems is poor retrieval. If the wrong chunks are retrieved, even a strong model will give a wrong or incomplete answer. Printing the retrieved sources before the answer is a simple way to debug this.