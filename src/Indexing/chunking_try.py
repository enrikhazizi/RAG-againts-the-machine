from pathlib import Path
import ast

my_python = Path("src/Indexing/chunking.py").resolve()

text = my_python.read_text(encoding="UTF-8")

tree = ast.parse(text)

for node in tree.body:
    print(node)