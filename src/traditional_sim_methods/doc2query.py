import torch
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
model.to(device)

corpus = pd.read_csv('../data/corpus_def.csv')

doct5_queries = []
for plain_text in corpus["plain_text"].values:
  if type(plain_text)==str:
    input_ids = tokenizer.encode(plain_text, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64, 
        num_return_sequences=1)

    doct5_queries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
  else:
    doct5_queries.append("")

aslett = pd.read_csv('../data/aslett_data.csv')
aslett = aslett[['ResponseId', 'URL', 'Category','FC_Eval', 'Search_Term', 'ALL_URLS', 'URLs', 'List_Scores', 'avg_score', 'query_length']]
smodel = SentenceTransformer("all-mpnet-base-v2")
sims = []
for url, gen_query in zip(corpus["URL"].values, doct5_queries):
  if gen_query:
    sims = []
    gen_query_emb = smodel.encode(gen_query)
    real_queries = aslett[aslett["URL"]==url]["Search_Term"].values
    real_queries_emb = smodel.encode(real_queries)
    similarity = smodel.similarity(gen_query_emb, real_queries_emb)
    sims.append(torch.mean(similarity).item())
print(np.mean(sims))