import chromadb
import logging
from rank_bm25 import BM25Okapi
import numpy as np

from embedding import E5LargeEmbeddingFunction

def get_window_range(num, window_range, doc_length):
    answer = [num]
    while len(answer) < window_range:
        f = 0
        left = answer[0] - 1
        right = answer[-1] + 1
        if left >= 0:
            answer = [left] + answer
        else:
            f += 1

        if right < doc_length:
            answer.append(right)
        else:
            f += 1

        if f == 2:
            return answer
    
    return answer

class ChromaDB:
    def __init__(self, host='db', port=8000, name="RAG_collection"):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.embedding_function = E5LargeEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(name=name, embedding_function=self.embedding_function)

        self.docs_counter = 10_001
        self.rows_counter = 0

        self.bm25_bag = None

    def insert_document(self, chunks, link=None, is_markdown=None, pages=None):
        self.embedding_function.change_mode(new_mode='passage')
        self.docs_counter += 1
        doc_id = self.docs_counter

        row_id = self.rows_counter

        self.collection.add(
            documents=chunks,
            metadatas=[{"link": link, "doc_id": doc_id, 'chunk_id': chunk_in_doc_id, 'max_length': len(chunks), 'is_markdown': is_markdown, 'pages': pages} 
                       for chunk_in_doc_id in range(len(chunks))],
            ids= ['v' + str(x) for x in  range(row_id, row_id+len(chunks))]
        )

        self.rows_counter += len(chunks)
        #print("done")

    def upsert_document(self, old_document_id, chunks, link=None, is_markdown=None, pages=None):
        self.embedding_function.change_mode(new_mode='passage')
            
        del_res = self.delete_document(old_document_id)
        if del_res is None:
            self.docs_counter += 1 #cuz -1 in delete_document
        else:
            print('nothing deleted, just insert') #nothing deleted

        row_id = self.rows_counter

        self.collection.add(
            documents=chunks,
            metadatas=[{"link": link, "doc_id": old_document_id, 'chunk_id': chunk_in_doc_id, 'max_length': len(chunks), 'is_markdown': is_markdown, 'pages': pages} 
                       for chunk_in_doc_id in range(len(chunks))],
            ids=list(map(str, range(row_id, row_id+len(chunks)))),
        )

        self.rows_counter += len(chunks)

    def reset_collection(self):
        elems = self.collection.get()
        if elems['ids']:
            self.collection.delete(elems['ids'])
            self.docs_counter = -1
            self.rows_counter = 0
            print('collection is empty')
        else:
            print('collection is already empty')

    def select_all(self):
        return self.collection.get()
    
    def get_chunk_by_ids(self, ids: list):
        return self.collection.get(ids)
    
    def delete_chunk_by_ids(self, ids: list):
        self.collection.delete(ids)

    def get_document(self, doc_id: int):
        return self.collection.get(where = {'doc_id': doc_id})
    
    def get_documents(self, doc_ids: list):
        return self.collection.get(where={"doc_id": {"$in": doc_ids}})
    
    def delete_document(self, doc_id: int):
        del_ids = self.get_document(doc_id)['ids']

        #assert len(del_ids) != 0, 'no documents found'
        if len(del_ids) == 0:
            return 'no documents found'
        
        self.collection.delete(del_ids)

        # self.rows_counter -= len(del_ids)
        self.docs_counter -= 1

    def delete_documents(self, doc_ids: list):
        del_ids = self.get_documents(doc_ids)['ids']

        #assert len(del_ids) != 0, 'no documents found'
        if len(del_ids) == 0:
            return 'no documents found'

        self.collection.delete(del_ids)

        # self.rows_counter -= len(del_ids)
        self.docs_counter -= len(doc_ids)

    def query(self, question, n_results=1, chunk_window = 3):
        '''chunk_window задаёт кол-во возвращаемых чанков'''

        self.embedding_function.change_mode(new_mode='query')

        query_res = self.collection.query(
            query_texts=[question],
            n_results=n_results)
        
        #assert query_res['ids'][0], 'query is empty'
        if not query_res['ids'][0]:
            return 

        most_relevant_doc_id = query_res['metadatas'][0][0]['doc_id']
        most_relevant_chunk_id = query_res['metadatas'][0][0]['chunk_id']
        doc_length = query_res['metadatas'][0][0]['max_length']

        window_chunks = [{"chunk_id": x} for x in get_window_range(most_relevant_chunk_id, chunk_window, doc_length)]

        return self.collection.get(where={"$and": [{"$or": window_chunks}, {"doc_id": most_relevant_doc_id}]})
    
    def query_knn(self, question, n_results=3, chunk_window = 3): 
        '''chunk_window задаёт кол-во возвращаемых чанков'''

        self.embedding_function.change_mode(new_mode='query')

        query_res = self.collection.query(
            query_texts=[question],
            n_results=n_results)
        
        #assert query_res['ids'][0], 'query is empty'
        if not query_res['ids'][0]:
            return 

        res = []
        for knn_res in query_res['metadatas'][0]:

            most_relevant_doc_id = knn_res['doc_id']
            most_relevant_chunk_id = knn_res['chunk_id']
            doc_length = knn_res['max_length']

            window_chunks = [{"chunk_id": x} for x in get_window_range(most_relevant_chunk_id, chunk_window, doc_length)]

            res.append(self.collection.get(where={"$and": [{"$or": window_chunks}, {"doc_id": most_relevant_doc_id}]}))

        return res
    
    def bm25_request(self, question, n_results=3, chunk_window=3):
        res = self.query_knn(question, n_results=n_results, chunk_window=chunk_window)

        all_links = []
        all_docs = []
        for i in res:
            all_links.append(i['metadatas'][0]['link'])
            all_docs.extend(i['documents'])

        tokenized_corpus = [doc.split(" ") for doc in all_docs]

        bm25 = BM25Okapi(tokenized_corpus)

        query = "как обратиться в банк"
        tokenized_query = query.split(" ")

        doc_scores = bm25.get_scores(tokenized_query)
        
        return res[np.argmax(doc_scores)//n_results]
