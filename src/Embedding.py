from src.settings import *

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # lut: look-up table
        # 将输入的词转换为词向量
        # vocab: 词表大小
        # d_model: 词向量的维度
        self.lut = nn.Embedding(vocab, d_model) 
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    