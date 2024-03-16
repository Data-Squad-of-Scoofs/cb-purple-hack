# 🥉 Hackathon of the Graduate School of Business and VK   
Predicting patient recovery using ML algorithms, image recognition for sanctuary by computer vision models, recommendation system for advertising banners 

## 🦸‍♂️ Team
We are the bronze medalists of this competition. 
Get to know us:
- [Solomon](https://github.com/veidlink)
- [Roman](https://github.com/gblssroman)
- [Nikita](https://github.com/qdzzzxc)
- [Vlad](https://github.com/vladik-pwnz)
- [Nikita](https://github.com/AnalyseOptimize)

# Навигация по репозиторию
- `dataset_preprocessing` - очистка корпуса спаршенных текстов
- `qa_generation` - генерация вопросов для валидации и finetun-a решения
- `knn_bm25.ipynb` - некоторые эксперименты с объединением двух алгоритмов поиска
- `chroma_db` - все файлы, связанные с запуском и наполнением ChromaDB
- `clickhouse` - запись данных в Clickhouse из csv файла
- `simularity.py` - скрипт по поиску релевантных документов
- `embeddings.py` - получение эмбендингов для БД
- `gui.ipynb` - **основной файл с фронтом приложения**
# Retrieval pipeline

<a href="https://ibb.co/hDGDr8L"><img src="https://i.ibb.co/0h0h1Jm/pipeline.jpg" alt="pipeline" border="0"></a>

# ***Как обращаться к LLM?***

поднимаем сервер:
1. поднимаем сервер в LM Studio
2. ```choco install ngrok```
3. ```ngrok config add-authtoken [authtoken]```
4. ```ngrok http --domain=live-relaxed-oryx.ngrok-free.app 1234 ```

обращаемся к LLM 
https://github.com/Data-Squad-of-Scoofs/cb-purple-hack/blob/db/inference_llm.ipynb

# ***Как обращаться к базе данных?***

Нами было реализовано для различных варианта. В итоге был выбран ClickHouse.

## 1. ChromaDB

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

## 2. ClickHouse

Запуск базы происходил через clickhouse.cloud, возможен также запуск через docker
Работает быстрее, чем ChromaDB

Подключение:

```python
pip install clickhouse_connect
client = clickhouse_connect.get_client(host=your_ip, port=your_port, username=your_user_name, password=your_password)
```

# Генерация валидационного датасета

<a href="https://ibb.co/TqnfPqk"><img src="https://i.ibb.co/QjqgYjf/14-03-2024-09-30-26.png" alt="14-03-2024-09-30-26" border="0"></a>


------

<a href="https://ibb.co/mvw67Gb"><img src="https://i.ibb.co/k3fQFx5/agents-d454d8169fbdc89ca73f7e23224a5122.png" alt="agents-d454d8169fbdc89ca73f7e23224a5122" border="0"></a>
