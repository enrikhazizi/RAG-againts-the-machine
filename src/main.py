from pathlib import Path
from Indexing.chunking import DatasetChunker

my_dir = Path("data/raw/vllm-0.10.1").resolve()
data_procesor = DatasetChunker()


data = data_procesor.chunk_directory(my_dir)
print(data)