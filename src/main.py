from chunking import chunker

with open("data/raw/vllm-0.10.1/setup.py", "r") as f:
    text = f.read()

splitter = chunker(max_tokens=100)
chunks = splitter.python_chunker(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")