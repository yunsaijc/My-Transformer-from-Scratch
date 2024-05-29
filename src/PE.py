from src.settings import *

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # max_len: 序列的最大长度
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # : - 所有行  0::2 - 从0开始的列，步长为2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape: (1, max_len, d_model)

        # 将pe注册为buffer，保存在model中，但不会被优化器更新
        # 保存需要在训练过程中保持不变，但又不应该被视为模型参数的变量
        # 可以通过 self.pe 来访问这个缓冲区
        self.register_buffer("pe", pe)  

    def forward(self, x):
        # x: shape: (batch_size, seq_len, d_model)
        # x.size(1): 输入的序列的长度。因为固定位置的 PE 是固定的，所以只需取前 x.size(1) 个位置的 PE
        # requires_grad_(False): 不需要计算梯度
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
        return self.dropout(x)
    

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))
    print("y.shape:", y.shape)
    print("y:", y)

    # 绘制前100个token位置的4个维度的 PE
    # 4个维度的 PE 的频率、offset不同
    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    chart = alt.Chart(data)\
            .mark_line()\
            .properties(width=800)\
            .encode(x="position", y="embedding", color="dimension:N")\
            .interactive()
    chart.save('/home/jc/workspace/My-Transformer-from-Scratch/figs/example_pe.html')

if __name__ == '__main__':
    example_positional()
