# Chunking and Embeddings Guide

## Why chunking matters
Documents are usually too long to embed as a single vector, and a single vector for a whole document blurs many topics together. Chunking splits documents into smaller passages so that each embedding represents a focused piece of information. Chunk size is a trade-off: small chunks (200-400 characters) give precise matches but lose context, while large chunks (1,500+ characters) keep context but dilute relevance.

## Chunking strategies
- Fixed-size chunking: split every N characters or tokens. Simple but may cut sentences in half.
- Recursive or structure-aware chunking: split on paragraphs first, then sentences, keeping chunks under a size limit. This is the strategy used in this project with a default chunk size of 800 characters.
- Semantic chunking: split where the topic changes, detected with embedding similarity between neighbouring sentences.
- Overlap: consecutive chunks share some text (this project uses 150 characters) so that facts that span a boundary are not lost.

## Text cleaning
Before chunking, text should be cleaned: normalising line endings, removing control characters, joining words hyphenated across line breaks, and collapsing repeated whitespace. PDF extraction in particular introduces broken lines and page artefacts.

## Embeddings
An embedding model maps text to a dense vector so that semantically similar texts are close together. This project uses the open-source sentence-transformers model all-MiniLM-L6-v2, which produces 384-dimensional vectors and runs well on a CPU. Vectors are L2-normalised so that cosine similarity equals the dot product. When the model is not available, the system falls back to TF-IDF vectors.

## Hybrid search
Dense (semantic) retrieval is good at paraphrases but can miss exact identifiers such as product codes, names or numbers. Sparse keyword retrieval with BM25 is the opposite. Hybrid search combines both: scores from each retriever are min-max normalised and mixed with a weight alpha (0.6 semantic and 0.4 keyword by default in this project). Hybrid search usually outperforms either retriever alone.

## Storage
The knowledge base stores, for every chunk, its text, the source document, its position and its vector. In this project the store is persisted to disk in the storage folder so that the index does not have to be rebuilt every time the app starts.
