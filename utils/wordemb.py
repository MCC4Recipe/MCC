# https://blog.csdn.net/IMchg/article/details/116161755

import torch
import random
from gensim.models import KeyedVectors


# # 加载词向量文件，文件里面包含了PAD、NONE的嵌入
# def load_embedding(wordemb_path):
#     word_embedding_dim = 300
#     word2idx = {}
#     wordemb = []
#     with open(wordemb_path, 'r', encoding='utf-8') as f:
#         index = 0
#         for line in f:
#             if index == 0:
#                 index += 1
#                 continue
#         for line in f:
#             splt = line.split()
#             assert len(splt) == word_embedding_dim+1
#             vector = list(map(float, splt[-word_embedding_dim:]))
#             word = splt[0]
#             word2idx[word] = len(word2idx)
#             wordemb.append(vector)
#
#     return word2idx, torch.DoubleTensor(wordemb)


# 加载词向量文件，文件里面不包含了PAD、NONE的嵌入
# 返回：word2idx,idx2word,wordemb
# idx2word这个其实没什么用，但是个人觉得方便检查数据，就加上了
def load_embedding(wordemb_path):
    word_embedding_dim = 300
    word2idx = {}
    wordemb = []
    # 原词向量中没有<PAD>、<UNK>，这里加上
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    vector_pad = [0.] * word_embedding_dim
    vector_unk = [random.random() for _ in range(word_embedding_dim)]
    wordemb.append(vector_pad)
    wordemb.append(vector_unk)

    with open(wordemb_path, 'r', encoding='utf-8') as f:
        index = 0
        for line in f:
            if index == 0:
                index += 1
                continue
            splt = line.split()
            assert len(splt) == word_embedding_dim + 1
            vector = list(map(float, splt[-word_embedding_dim:]))
            word = splt[0]
            word2idx[word] = len(word2idx)
            wordemb.append(vector)

    return word2idx, torch.DoubleTensor(wordemb)
