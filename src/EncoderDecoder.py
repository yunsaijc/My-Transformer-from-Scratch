from src.settings import *

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # source embedding
        self.tgt_embed = tgt_embed  # target embedding
        self.generator = generator  # ?? TODO: find out what this is

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Combine self.encoder and self.decoder"
        "Take in and process masked src and target sequences."
        "tgt 是decoder的输出"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory: the output of the encoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        # super() 用于调用Generator父类 nn.Module 的 __init__() 方法 
        super(Generator, self).__init__()
        # 定义一个全连接的线性层，输入维度为d_model，输出维度为vocab
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
