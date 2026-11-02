# Embeddings

An embedding is a list of numbers that represents the meaning of a piece of text. Texts with similar meanings produce embeddings that are close together, even if they use completely different words. For example, "car" and "automobile" have very similar embeddings.

Embeddings are created by embedding models. A popular small model is all-MiniLM-L6-v2, which turns any sentence into a vector of 384 numbers. It is fast enough to run on a regular laptop without a GPU.

Similarity between two embeddings is usually measured with cosine similarity. Cosine similarity looks at the angle between two vectors. A score close to 1 means the texts are very similar, and a score close to 0 means they are unrelated.

Cosine distance is simply 1 minus cosine similarity. Vector databases often report distance instead of similarity, so a lower distance means a better match.

The same embedding model must be used for both the documents and the search query. If you embed documents with one model and queries with another, the numbers will not be comparable and search results will be meaningless.