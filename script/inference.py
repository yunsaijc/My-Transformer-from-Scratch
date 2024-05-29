from src.Transformer import *
from src.Decoder import subsequent_mask

def inference_test():
    # src_vocab = 11, tgt_vocab = 11, N = 2 (number of layers)
    test_model = make_model(11, 11, 2)
    test_model.eval()

    # src.shape = (1, 10), src_mask.shape = (1, 1, 10)
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    # ys即tgt, 为decoder的输出
    ys = torch.zeros(1, 1).type_as(src)

    # 因为输入序列长度为10，所以需要对剩下的9个位置进行预测
    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat( # 将预测的单词拼接到ys后面
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)

def run_tests():
    for _ in range(10):
        inference_test()

if __name__ == '__main__':
    run_tests()
