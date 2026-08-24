from Indexing.chunking import chunker
import bm25s


with open("data/raw/vllm-0.10.1/setup.py", "r") as f:
    text = f.read()


splitter = chunker(max_tokens=100)
chunks = splitter.python_chunker(text)


corpus_tokens = bm25s.tokenize(chunks, stopwords="en")

retriever = bm25s.BM25()
retriever.index(corpus_tokens)

 
query = "what does the load_module_from_path do?"
query_tokens = bm25s.tokenize(query)
results, scores = retriever.retrieve(query_tokens, k=2)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")
