from src.settings import *
from src.Global import *

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # 计算query的最后一个维度的大小，即embedding的维度
    d_k = query.size(-1) 

    # key.transpose(-2, -1)表示将key的倒数第二个维度和倒数第一个维度交换
    # 即将 seq_len * d_k 的矩阵做转置，方便矩阵相乘
    # scores表示注意力得分矩阵
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # mask == 0产生一个布尔值的矩阵
        # 为True的位置用-1e9填充，在softmax之后对应位置的值就为0
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # p_attn是attention的概率分布
    p_attn = scores.softmax(dim=-1) 
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # 每个头的维度
        self.h = h  # 多头注意力的头数
        # 4个线性变换，分别对应q, k, v和最后的输出
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # 在第1维增加一个维度
        nbatches = query.size(0) # batch_size

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 将query, key, value分别通过线性变换得到q, k, v
        # 并将从 batch_size * seq_len * d_model 
        # 变换为 batch_size * h * seq_len * d_k
        # 注意区分：seq_len是序列长度，d_model是序列中每一个token的embedding维度
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)   # batch_size * h * seq_len * d_k => batch_size * seq_len * h * d_k
            .contiguous()   # 使得x的存储在内存中是连续的，在transpose之后需要调用contiguous，从而可以调用view
            .view(nbatches, -1, self.h * self.d_k)  # 变回原来的维度：batch_size * seq_len * d_model
        )
        del query
        del key
        del value
        return self.linears[-1](x) # 最后的线性变换
