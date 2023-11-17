import torch
from torch import nn
# 定义编码器，词典大小为10，要把token编码成128维的向量
embedding = nn.Embedding(10, 128)
# 定义transformer，模型维度为128（也就是词向量的维度）
transformer = nn.Transformer(d_model=128, batch_first=True) # batch_first一定不要忘记
# 定义源句子，可以想想成是 <bos> 我 爱 吃 肉 和 菜 <eos> <pad> <pad>
src = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2, 2]])
# 定义目标句子，可以想想是 <bos> I like eat meat and vegetables <eos> <pad>
tgt = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2]])
# 将token编码后送给transformer（这里暂时不加Positional Encoding）
print(embedding(src).shape, embedding(tgt).shape)
outputs = transformer(embedding(src), embedding(tgt))
print(outputs.shape)