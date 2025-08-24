# from torchtext import vocab
# import torch
# import torch.nn as nn
# import pandas as pd
# import polars as pl
# import pickle

# class GLOVE(object):
#     def __init__(self, ):
#         self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
#         self.vocab.itos.extend(['<unk>'])
#         self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
#         self.vocab.vectors = torch.cat(
#             (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
#         self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)

#     def get_query(self,query):
#         word_inds = torch.LongTensor(
#             [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
#         return self.embedding(word_inds)
    
# df = pl.read_csv("./sampled_data.csv")
# # df = pd.read_csv('./sampled_data.csv')

# # Group by consumer and aggregate the 'store' and 'duration_in_sec' columns into lists
# aggregated_df = df.groupby('pid_id').agg(pl.col('store_duration').alias('store_list'))
# # Filter out rows where store_sequence has only 2 elements and they are the same
# # aggregated_df = aggregated_df.filter(~((aggregated_df["store_list"].len() == 3) & (aggregated_df["store_list"].apply(lambda x: x[0] == x[1]))))

# # Convert store sequences into a list of lists for Word2Vec training
# sentences = aggregated_df["store_list"].to_list()

# glove = GLOVE()

# embedding_dict = {}
# for i in range(len(sentences)):
#     for j in range(len(sentences[i])):
#         # term = sentences[i][j]
#         term = sentences[i][j].replace('_', ' ')
#         if term not in embedding_dict:  # check if the embedding vector exists in the dict
#             embedding_dict[term] = glove.get_query(term).detach().numpy()
#             # if u want to store the embedding vectors as tensors in the dictionary, you can directly store them
#             # if u want to store the embedding vectors as NumPy arrays, you can use the .detach().numpy() method

# filename = 'embedding_dict.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump(embedding_dict, f)
# print(f"Embedding dictionary saved to {filename}")

# import numpy as np
#  # Split each (store, duration_bucket) pair to get the store name
# # store_names = [pair.rsplit(' ', 1)[0] for pair in embedding_dict.keys()]

# aggregated_embeddings = {}
# for pair in embedding_dict.keys():
#     store_name = pair.rsplit(' ', 1)[0]
#     aggregated_embeddings[store_name] = np.mean(embedding_dict[pair][:-1], axis=0)
#     # Extract all store embeddings into a list and maintain their order
#     stores = list(aggregated_embeddings.keys())
#     embeddings_list = [aggregated_embeddings[store] for store in stores]

#     embeddings_dict = dict(zip(stores, embeddings_list))


# # # Aggregate embeddings for each store
# # for store in set(store_names):
# #     store_pairs = [pair for pair in embedding_dict.keys()]
# #     print(np.mean(embedding_dict[store_pairs[0]][:-1],axis=0).shape)
# #     # Average the embeddings of all pairs for this store
# #     aggregated_embeddings[store] = np.mean([embedding_dict[pair][:-1] for pair in store_pairs], axis=0)

# #     # # Extract all store embeddings into a list and maintain their order
# #     # stores = list(aggregated_embeddings.keys())
# #     # embeddings_list = [aggregated_embeddings[store] for store in stores]

# #     # embeddings_dict = dict(zip(stores, embeddings_list))


# from __future__ import print_function
from torchtext import vocab
import torch
import torch.nn as nn
import pandas as pd
import polars as pl
import pickle

from glove import Glove
from glove import Corpus






class GLOVE(object):
    def __init__(self, ):
        self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
        self.vocab.itos.extend(['<unk>'])
        self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
        self.vocab.vectors = torch.cat(
            (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
        self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)

    def get_query(self,query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)
    
df = pl.read_csv("./sampled_data.csv")
# df = pd.read_csv('./sampled_data.csv')

# Group by consumer and aggregate the 'store' and 'duration_in_sec' columns into lists
aggregated_df = df.groupby('pid_id').agg(pl.col('store_duration').alias('store_list'))
# Filter out rows where store_sequence has only 2 elements and they are the same
# aggregated_df = aggregated_df.filter(~((aggregated_df["store_list"].len() == 3) & (aggregated_df["store_list"].apply(lambda x: x[0] == x[1]))))

# Convert store sequences into a list of lists for Word2Vec training
sentences = aggregated_df["store_list"].to_list()
print(sentences)

corpus_model = Corpus()
corpus_model.fit(sentences, window=10)
#corpus_model.save('corpus.model')
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)
# train------no_components: data dimension,u can diy
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)
# # save model
corpus_model.save('corpus.model')
# corpus_model = Corpus.load('corpus.model')
glove.save('glove.model')

# # load model
glove = Glove.load('glove.model')
# 全部词向量矩阵
# print(glove.word_vectors.shape)
# 指定词条词向量
# print(glove.word_vectors[glove.dictionary['Miniso _Q4']].shape)
print(glove.most_similar('Baoshifu_Q2', number=10))

embedding_dict = {}
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        term = sentences[i][j]
        if term not in embedding_dict:  # check if the embedding vector exists in the dict
            embedding_dict[term] = glove.word_vectors[glove.dictionary[term]]
            # if u want to store the embedding vectors as tensors in the dictionary, you can directly store them
            # if u want to store the embedding vectors as NumPy arrays, you can use the .detach().numpy() method
filename = 'embedding_dict.pkl'
with open(filename, 'wb') as f:
    pickle.dump(embedding_dict, f)
print(f"Embedding dictionary saved to {filename}")

import numpy as np
#  # Split each (store, duration_bucket) pair to get the store name
# # store_names = [pair.rsplit(' ', 1)[0] for pair in embedding_dict.keys()]
from collections import defaultdict

tmp_embeddings = defaultdict(list)
for pair in embedding_dict.keys():
    store_name = pair.rsplit('_', 1)[0]
    for j in range(1,5):
        if store_name+'_Q'+str(j) in embedding_dict.keys():
            # print(embedding_dict[store_name+'_Q'+str(j)])
            # print(glove.dictionary[store_name+'_Q'+str(j)].shape)
            tmp_embeddings[store_name].append(embedding_dict[store_name+'_Q'+str(j)])
                    
aggregated_embeddings={}
for store_name in tmp_embeddings.keys():
    aggregated_embeddings[store_name] = np.mean(tmp_embeddings[store_name][:],axis=0)

#     # Extract all store embeddings into a list and maintain their order
#     stores = list(aggregated_embeddings.keys())
#     embeddings_list = [aggregated_embeddings[store] for store in stores]

#     embeddings_dict = dict(zip(stores, embeddings_list))

