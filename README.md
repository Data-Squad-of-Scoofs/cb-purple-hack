# Навигация по репозиторию
- `eda_from_corpus.ipynb` - очистка корпуса спаршенных текстов
- `qa_generation` - генерация вопросов для валидации и finetun-a решения
- `knn_bm25.ipynb` - некоторые эксперименты с объединением двух алгоритмов поиска
- `chroma_db.py` - скрипт, настраивающий ChromaDB
- `click.ipynb` - запись данных в Clickhouse из csv файла
- `simularity.py` - скрипт по поиску релевантных документов
- `embeddings.py` - получение эмбендингов для БД

# ***Как обращаться к LLM?***

поднимаем сервер:
1. поднимаем сервер в LM Studio
2. ```choco install ngrok```
3. ```ngrok config add-authtoken [authtoken]```
4. ```ngrok http --domain=live-relaxed-oryx.ngrok-free.app 1234 ```

обращаемся к LLM 
https://github.com/Data-Squad-of-Scoofs/cb-purple-hack/blob/db/inference_llm.ipynb

# ***Как обращаться к базе данных?***

нами было реализовано для различных варианта

## I ChromaDB

Запуск базы в docker:
```bash
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```

Подключение:

```python
pip install chromadb
self.client = chromadb.HttpClient(host=your_ip, port=your_port)
```

## II ClickHouse

Запуск базы происходил через clickhouse.cloud, возможен также запуск через docker

Подключение:

```python
pip install clickhouse_connect
client = clickhouse_connect.get_client(host=your_ip, port=your_port, username=your_user_name, password=your_password)
```
