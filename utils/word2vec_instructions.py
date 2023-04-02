from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# f = open('../data/tokenized_instructions_train.txt', 'r')
# text = f.read()
# f.close()
# f = open('../data/tokenized_instructions_test.txt', 'r')
# text += f.read()
# f.close()
# f = open('../data/tokenized_instructions_val.txt', 'r')
# text += f.read()
# f.close()
#
# f = open('../data/tokenized_instructions.txt', 'w')
# f.write(text)
# f.close()

model = Word2Vec(
    LineSentence(open('../data/tokenized_instructions.txt', 'r', encoding='utf8')),
    sg=0,
    size=300,
    window=5,
    min_count=1
)

# 词向量保存
model.wv.save_word2vec_format('instr.vec', binary=False)

# 模型保存
model.save('w2v_instr.model')
