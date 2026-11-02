# Vector Databases

A vector database stores embeddings and makes it fast to find the vectors closest to a query. Regular databases search for exact matches, while vector databases search for meaning.

ChromaDB is an open-source vector database that is popular for learning and prototyping. It can run entirely in memory or save data to disk with a PersistentClient. By default, ChromaDB embeds documents automatically using the all-MiniLM-L6-v2 model.

Data in ChromaDB is organized into collections. Each item in a collection has a unique ID, the document text, its embedding, and optional metadata such as the source file name.

Metadata filtering lets you narrow a search before comparing vectors. For example, you can search only chunks where the source is a specific file. This is useful when users want answers from one document.

Most vector databases use an algorithm called HNSW, which stands for Hierarchical Navigable Small World. HNSW finds approximate nearest neighbors very quickly, even in collections with millions of vectors, by building a layered graph of connections between similar items.