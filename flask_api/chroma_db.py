import chromadb
import logging

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
        logging.info("check init")
        self.client = chromadb.HttpClient(host=host, port=port)
        logging.info("check client")
        self.embedding_function = E5LargeEmbeddingFunction()
        logging.info("check emb")
        self.collection = self.client.get_or_create_collection(name=name, embedding_function=self.embedding_function)
        logging.info("connected to collection")

        self.docs_counter = -1 #maximum doc id
        all_rows = self.collection.get()
        for row in all_rows['metadatas']:
            self.docs_counter =  max(self.docs_counter, row['doc_id'])

        if all_rows['ids']:
            self.rows_counter = int(max(all_rows['ids'])) + 1
        else:
            self.rows_counter = 0

    def insert_document(self, chunks, sources=None, document_type=None, pages=None):
        if document_type is None:
            document_type = 'markdown'

        # assert len(chunks)  == len(sources)
        if len(chunks) != len(sources):
            return 'Each chunk must have one source'

        if document_type == 'pdf':
            #assert len(chunks)  == len(sources) == len(pages)
            if pages is None or len(chunks) != len(pages):
                return 'Each PDF chunk must have one page number'  
        elif document_type == 'markdown' and pages is None:
            pages = [1] * len(chunks)

        self.docs_counter += 1
        doc_id = self.docs_counter

        row_id = self.rows_counter

        self.collection.add(
            documents=chunks,
            metadatas=[{"source": source, "doc_id": doc_id, 'chunk_id': chunk_in_doc_id, 'max_length': len(chunks), 'doc_type': document_type, 'page': page} 
                       for chunk_in_doc_id, source, page in zip(range(len(sources)), sources, pages)],
            ids=list(map(str, range(row_id, row_id+len(chunks)))),
        )

        self.rows_counter += len(chunks)

    def upsert_document(self, old_document_id, chunks, sources=None, document_type=None, pages=None):
        if document_type is None:
            document_type = 'markdown'

        #assert len(chunks)  == len(sources)
        if len(chunks) != len(sources):
            return 'Each chunk must have one source'

        if document_type == 'pdf':
            #assert len(chunks)  == len(sources) == len(pages)
            if pages is None or len(chunks) != len(pages):
                return 'Each PDF chunk must have one page number'
            
        elif document_type == 'markdown' and pages is None:
            pages = [1] * len(chunks)

        # try:
        #     self.delete_document(old_document_id)

        #     self.docs_counter += 1 #cuz -1 in delete_document
        # except AssertionError:
        #     print('just insert')
            
        del_res = self.delete_document(old_document_id)
        if del_res is None:
            self.docs_counter += 1 #cuz -1 in delete_document
        else:
            print('just insert') #nothing deleted

        row_id = self.rows_counter

        self.collection.add(
            documents=chunks,
            metadatas=[{"source": source, "doc_id": old_document_id, 'chunk_id': chunk_in_doc_id, 'max_length': len(chunks), 'doc_type': document_type, 'page': page} 
                       for chunk_in_doc_id, source, page in zip(range(len(sources)), sources, pages)],
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
        logging.info("all docs selected")
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
    
# chroma = ChromaDB()
# print(chroma.select_all())

# chroma.reset_collection()
# print(chroma.rows_counter)
# print(chroma.docs_counter)

# chroma.insert_document(chunks=['123', 'fdg', 'fdsgeg'], sources=['1', '2', '3'])
# chroma.insert_document(chunks=['what is this doc', 'hello world', 'как ты'], sources=['1', '2', '3'])

# # print(chroma.select_all())
# # print(chroma.docs_counter)

# print(chroma.delete_document(1))

# # #print(chroma.select_all())
# # #chroma.upsert_document(old_document_id=0, chunks=['test1', 'test2'], sources=['1', '2'])
# chroma.upsert_document(old_document_id=2, chunks=['popa2'], sources=['2131'])
