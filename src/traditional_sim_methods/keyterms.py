import torch
from keybert import KeyBERT
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

kw_model = KeyBERT()
corpus = pd.read_csv('../data/corpus_def.csv')

keybert = []
for plain_text in corpus["plain_text"].values:
  if type(plain_text)==str:
    keywords = kw_model.extract_keywords(plain_text)
    query=" ".join([k for k,v in keywords])

    keybert.append(query)
  else:
    keybert.append("")

aslett = pd.read_csv('../data/aslett_data.csv')
aslett = aslett[['ResponseId', 'URL', 'Category','FC_Eval', 'Search_Term', 'ALL_URLS', 'URLs', 'List_Scores', 'avg_score', 'query_length']]
smodel = SentenceTransformer("all-mpnet-base-v2")
sims = []
for url, gen_query in zip(corpus["URL"].values, keybert):
  if gen_query:
    sims = []
    gen_query_emb = smodel.encode(gen_query)
    real_queries = aslett[aslett["URL"]==url]["Search_Term"].values
    real_queries_emb = smodel.encode(real_queries)
    similarity = smodel.similarity(gen_query_emb, real_queries_emb)
    sims.append(torch.mean(similarity).item())
print(np.mean(sims))