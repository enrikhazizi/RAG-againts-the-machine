from pathlib import Path
from Indexing.chunking import DatasetChunker
from vllm import LLM, SamplingParams
import bm25s

llm = LLM(model="Qwen/Qwen3-0.6B")
sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

my_dir = Path("data/raw/vllm-0.10.1").resolve()
data_procesor = DatasetChunker()


corpus = data_procesor.chunk_directory(my_dir)
corpus_tokens = bm25s.tokenize(
    [chunk["text"] for chunk in corpus],
    stopwords=None
)

retriever = bm25s.BM25(corpus=corpus)
retriever.index(corpus_tokens)

query = "how does vllm serve openai chat?"
query_tokens = bm25s.tokenize(query)
results, scores = retriever.retrieve(query_tokens, k=5)

prompt = f"""Use the following document context to answer the user question.
Context:
{results}

Question: {query}
Answer:"""

retriever.save("data/proccessed/vllm")
outputs = llm.generate(prompt, sampling_params)

# 4. Print the generated text
for output in outputs:
    print(output.outputs[0].text)
