from settings import *
from Global import *
from utils import *

"""
(Transformer Paper)
Decoder = N * Layer
Layer = Sublayer1 + Sublayer2 + Sublayer3
      = Masked Self-Attention + Add&Norm (Sublayer1)
        + Attention + Add&Norm (Sublayer2)
        + Feed-Forward + Add&Norm (Sublayer3)

(This Code)
Encoder -> EncoderLayer -> SublayerConnection
Decoder = N * DecoderLayer
DecoderLayer = Sublayer1 + Sublayer2 + Sublayer3
             = self_attn(masked) + SublayerConnection (Sublayer1)
               + src_attn + SublayerConnection (Sublayer2)
               + feed_forward + SublayerConnection (Sublayer3)
"""

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory  # memory是encoder的输出
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # src_attn的K和V是encoder的输出, Q是decoder的输出
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) 
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    "掩盖后续位置的元素"
    attn_shape = (1, size, size)
    # torch.triu返回矩阵的上三角部分, diagonal=1表示对角线中的元素包含在上三角部分中
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0 # 上三角部分为0, 下三角部分为1

def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )
    chart = alt.Chart(LS_data)\
            .mark_rect()\
            .properties(height=250, width=250)\
            .encode(
                alt.X("Window:O"),
                alt.Y("Masking:O"),
                alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
            )\
            .interactive()
    chart.save('/home/jc/workspace/My-Transformer-from-Scratch/figs/example_mask.html')

if __name__ == '__main__':
    example_mask()

    # print(subsequent_mask(3))
    # tensor([[
    #   [True, False, False],
    #   [True, True, False],
    #   [True, True, True]]])
