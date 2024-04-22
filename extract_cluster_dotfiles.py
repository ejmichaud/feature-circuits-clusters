import gzip
import io
import os
from sqlitedict import SqliteDict

# list directories in the data folder
dataset_names = os.listdir("data")
dataset_names = [name for name in dataset_names if os.path.isdir(f"data/{name}")]

# mkdir dotfiles
if not os.path.exists("dotfiles"):
    os.mkdir("dotfiles")

for name in dataset_names:
    if not os.path.exists(f"dotfiles/{name}"):
        os.mkdir(f"dotfiles/{name}")
    if os.path.exists(f"data/{name}/circuit_graphviz.sqlite"):
        with SqliteDict(f"data/{name}/circuit_graphviz.sqlite") as db:
            for cluster_idx, compressed_bytes in db.items():
                decompressed_object = io.BytesIO(compressed_bytes)
                with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
                    with open(f"dotfiles/{name}/{cluster_idx}.dot", "wb") as f:
                        for i, line in enumerate(file.readlines()):
                            if i == 0:
                                line = b'digraph "Feature circuit" {'
                            if line == b'\x94s.':
                                break
                            f.write(line)
    else:
        print("No circuit_graphviz.sqlite file found in the selected database.")