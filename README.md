# 🤖 IT Purple Hack & Central Bank of Russia
Building LLM-based RAG chat-bot for the Central Bank.

## 🦸‍♂️ Team
We are the finalists of this competition. 
Get to know us:
- [Solomon](https://github.com/veidlink)
- [Roman](https://github.com/gblssroman)
- [Nikita](https://github.com/qdzzzxc)
- [Vlad](https://github.com/vladik-pwnz)
- [Nikita](https://github.com/AnalyseOptimize)

## ***Navigation***
- `dataset_preprocessing` - cleaning the corpus of scraped texts
- `qa_generation` - generating questions for validation and finetuning the solution
- `knn_bm25.ipynb` - some experiments with combining two search algorithms
- `chroma_db` - all files related to launching and populating ChromaDB
- `clickhouse` - writing data to Clickhouse from a csv file
- `simularity.py` - script for finding relevant documents
- `embeddings.py` - obtaining embeddings for the database
- `gui.ipynb` - **the main file with the application front-end**
# Retrieval pipeline

[![pipeline](https://i.ibb.co/0h0h1Jm/pipeline.jpg)](https://ibb.co/hDGDr8L)

# ***How to interact with LLM?***

Starting the server:
1. Launch the server in LM Studio
2. ```choco install ngrok```
3. ```ngrok config add-authtoken [authtoken]```
4. ```ngrok http --domain=live-relaxed-oryx.ngrok-free.app 1234 ```

Interacting with LLM 
https://github.com/Data-Squad-of-Scoofs/cb-purple-hack/blob/db/inference_llm.ipynb

# ***How to interact with the database?***

We implemented various options. Eventually, ClickHouse was chosen.

## 1. ChromaDB

Launching the database in docker:
```
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```

Connecting:

```
pip install chromadb
self.client = chromadb.HttpClient(host=your_ip, port=your_port)
```

## 2. ClickHouse

The database was launched via clickhouse.cloud, but it can also be launched via docker.
It operates faster than ChromaDB.

Connecting:

```
pip install clickhouse_connect
client = clickhouse_connect.get_client(host=your_ip, port=your_port, username=your_user_name, password=your_password)
```

# Generating a validation dataset

[![14-03-2024-09-30-26](https://i.ibb.co/QjqgYjf/14-03-2024-09-30-26.png)](https://ibb.co/TqnfPqk)

------

[![agents-d454d8169fbdc89ca73f7e23224a5122](https://i.ibb.co/k3fQFx5/agents-d454d8169fbdc89ca73f7e23224a5122.png)](https://ibb.co/mvw67Gb)
