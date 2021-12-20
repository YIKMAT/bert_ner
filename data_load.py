'''
An entry or sent looks like ...

SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O

Each mini-batch returns the followings:
words: list of input sents. ["The 26-year-old ...", ...]
x: encoded input sents. [N, T]. int64.
is_heads: list of head markers. [[1, 1, 0, ...], [...]]
tags: list of tags.['O O B-MISC ...', '...']
y: encoded tags. [N, T]. int64
seqlens: list of seqlens. [45, 49, 10, 50, ...]
'''
import numpy as np
import torch
from torch.utils import data

from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class NerDataset(data.Dataset):
    def __init__(self, fpath):
        """
        fpath: [train|valid|test].txt
        """
        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents, tags_li = [], [] # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)


    # 函数功能： 获取样本。
    # 输入： idx是样本序号
    # 输出： len(words) = len(tags)  # : len(x)==len(y)==len(is_heads)==seqlen
    # words  原始的单词，未经过tokenizer
    # x: 经过tokenizer, token对应的id
    # is_heads: 标识符，token是否有效(经过tokenizer的第一个单词)
    # tags: word的NER标签 (tokenizer之前的)
    # y: 经过tokenizer,token的NER标签id
    # seqlen: seq中token的长度
    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        # 因为有些单词经过tokenizer以后，会变成好几个单词(也就是token)，所以：此处是只算第一个单词，其他的算<pad>
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        # 断言：如果不成立，则执行后面的语句  f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}
        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)

        return words, x, is_heads, tags, y, seqlen

# 函数功能： 给每个batch中的所有样本，赋予max_seq_length，不够的增加<PAD>
# 输入： batch_size个样本 tuple(batch_size), 每个样本有6个信息，分别是words, x, is_heads, tags, y, seqlen
# 输出： words, x(已按max_sequence_len增加<PAD>), is_heads, tags, y(已按max_sequence_len增加<PAD>), seqlen
def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens


