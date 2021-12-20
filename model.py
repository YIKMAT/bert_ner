import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.fc = nn.Linear(768, vocab_size)

        self.device = device
        self.finetuning = finetuning


    # 功能：调用模型
    # 输入： x为token，shape(batch_size, max_seq_len)   y为tag，shape(batch_size, max_seq_len)
    # 输出： logits为分类的结果，shape(batch_size, max_seq_len, tag_num)
    #          y为真实tag，shape(batch_size, max_seq_len)
    #          y为预测tag，shape(batch_size, max_seq_len)
    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            # feature_based  直接使用bert的结果，不微调
            self.bert.eval()
            # with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源
            # torch.no_grad()：在该模块下，所有计算得出的tensor的requires_grad都自动设置为False，不用自动求导了
            with torch.no_grad():
                # encoded_layers是所有层（这里是12层）的隐状态输出
                encoded_layers, _ = self.bert(x)
                # 最后一层的隐状态输出，shape(batch_size, max_seq_len, bert输出维度【768】)
                enc = encoded_layers[-1]

        if self.top_rnns:
            #
            enc, _ = self.rnn(enc)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

