from embedding import E5LargeEmbeddingFunction
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import nltk
from nltk.tokenize import word_tokenize
import clickhouse_connect
import pandas as pd
import num2text
import numpy as np
from rank_bm25 import BM25Okapi

model = AutoModelForSequenceClassification.from_pretrained('SkolkovoInstitute/ruRoberta-large-paraphrase-v1')
tokenizer = AutoTokenizer.from_pretrained('SkolkovoInstitute/ruRoberta-large-paraphrase-v1')

def prep_query(query):
    temp = [num2text.num2text(int(word)) if word.isdigit() else word for word in nltk.word_tokenize(query, language='russian')]

    sentence = ''
    for word in temp:
        if word not in ',,.?!:)»':
            sentence += " " + word

    return sentence

def get_window_range(num, window_range):
    answer = [num]
    while len(answer) < window_range:
        f = 0
        left = answer[0] - 1
        right = answer[-1] + 1

        if left >= 0:
            answer = [left] + answer
            if len(answer) >= window_range:
                return answer

        answer.append(right)
    
    return answer

def _clickhouse_query_l2(query, client, emb_func, limit=5, table='index_texts_final'):
    emb_func.change_mode('query')
    embeddings = emb_func(query)[0]

    result = client.query(f'''SELECT
        ID, chunk_id,
        text,
        L2Distance(embedding, {embeddings}) AS score
    FROM {table}
    ORDER BY score ASC
    LIMIT {limit}''')

    return result.result_rows

def _clickhouse_query_window(query, client, emb_func, limit_knn=5, docs_window=5, table='index_texts_final'):
    res = _clickhouse_query_l2(query, client, emb_func, limit=limit_knn, table=table)

    window_sql_query = ''
    for i, row in enumerate(res):
        if i > 0:
            window_sql_query += ' UNION DISTINCT '

        window_sql_query += f'''SELECT * FROM {table} WHERE ID  in {tuple(get_window_range(row[0], docs_window))}'''

    result = client.query(window_sql_query)
    return result.result_rows

def get_similarity(text1, text2):
    '''Cross-Encoder similarity thanks to Skolkovo fine-tuners'''
    with torch.inference_mode():
        batch = tokenizer(
            text1, text2, 
            truncation=True, max_length=model.config.max_position_embeddings, return_tensors='pt',
        ).to(model.device)
        proba = torch.softmax(model(**batch).logits, -1)
    return proba[0][1].item()

def bm25_ensemble_with_crossenc_answer(query, client, emb_func, bm25_n_results=10, cr_enc_n_results=2, limit_knn=70, knn_docs_window=3):
    res = _clickhouse_query_window(query, client, emb_func, limit_knn=limit_knn, docs_window=knn_docs_window)

    all_links = []
    all_docs = []
    all_pages = []

    for row in res:
        all_links.append(row[2])
        all_docs.append(row[3])
        all_pages.append(row[4])

    tokenized_corpus = [word_tokenize(doc, language='russian') for doc in all_docs]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = word_tokenize(query, language='russian')

    doc_scores = bm25.get_scores(tokenized_query)

    if all(doc_scores == 0):
        return ['Все найденные через KNN документы не имеют ничего общего к запросу по мнению bm25']
    
    bm25_answer = []
    args = np.argsort(doc_scores, axis=0)

    for i in range(1, bm25_n_results+1):
        bm25_answer.append(res[args[-i]])
        
    crossenc_answer = []
    for p in range(bm25_n_results):
        crossenc_answer += [get_similarity(query, bm25_answer[p][3])]
    

    final_ans = []
    args_cr = np.flipud(np.argsort(crossenc_answer, axis=0))

    for f in range(cr_enc_n_results):
        final_ans.append(bm25_answer[args_cr[f]])
    
    return final_ans