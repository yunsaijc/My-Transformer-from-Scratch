
from settings import *
from Global import *

"""
(Transformer Paper)
Encoder = N * Layer
Layer = Sublayer1 + Sublayer2
      = Self-Attention + Add&Norm (Sublayer1)
        + Feed-Forward + Add&Norm (Sublayer2)

(This Code)
Encoder -> EncoderLayer -> SublayerConnection
Encoder = N * EncoderLayer
EncoderLayer = Sublayer1 + Sublayer2
             = self_attn + SublayerConnection (Sublayer1)
               + feed_forward + SublayerConnection (Sublayer2)
"""


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:   # layers的堆叠方式是串联的
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # sublayer[0]是attn后的 Add&Norm，sublayer[1]是FFN后的 Add&Norm
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
