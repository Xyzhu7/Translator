# -*- coding: UTF-8 -*-
import jieba
import torch
from utils import normalizeString,cht_to_chs,tensorsFromPair
from sklearn.model_selection import train_test_split
from configs import Config



class Lang_Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "UNK", 2: "SOS", 3: "EOS"}
        self.n_words = 4

    def addWord(self, word):  # 构建词表
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):  # 按句子构建
        for word in sentence.split(" "):
            self.addWord(word)


def readLangs(lang1, lang2, path, config):  # 将原始数据转变为符合Lang类的形式
    lines = open(path, encoding="utf-8").readlines()
    lang1_cls = Lang_Vocab(lang1)
    lang2_cls = Lang_Vocab(lang2)
    pairs = []
    for l in lines:
        l = l.split("\t")
        sentence1 = normalizeString(l[0])
        sentence2 = cht_to_chs(l[1])
        seg_list = jieba.cut(sentence2, cut_all=False)
        sentence2 = " ".join(seg_list)
        if ((len(sentence1.split(" ")) > config.maxlength) or
                (len(sentence2.split(" ")) > config.maxlength)):  # 大于maxlength的舍去
            continue
        else:
            pairs.append([sentence1, sentence2])
            lang1_cls.addSentence(sentence1)
            lang2_cls.addSentence(sentence2)

    return lang1_cls, lang2_cls, pairs


def splitData(pairs, test_ratio, random_state):
    pairs_train = []
    pairs_test = []
    pairs_train, pairs_test = (train_test_split(pairs, test_size=test_ratio, random_state=random_state))
    return pairs_train, pairs_test


lang1 = "en"
lang2 = "cn"
path = "./sources/cmn.txt"
config = Config()
lang1_cls, lang2_cls, pairs = readLangs(lang1, lang2, path, config)
pairs_train, pairs_test = splitData(pairs, test_ratio=0.2, random_state=42)

def pairsToTensors(pairs):
    pairs_en = torch.zeros((len(pairs), config.maxlength), dtype=torch.long)
    pairs_ch = torch.zeros((len(pairs), config.maxlength), dtype=torch.long)
    for idx, pair in enumerate(pairs):
        pair_en, pair_ch = tensorsFromPair(lang1_cls, lang2_cls, pair)
        pair_en = torch.nn.functional.pad(pair_en, (0, config.maxlength - len(pair_en)))
        pair_ch = torch.nn.functional.pad(pair_ch, (0, config.maxlength - len(pair_ch)))
        pairs_en[idx] = pair_en
        pairs_ch[idx] = pair_ch  # 把数据处理成pairs_train_en[idx]和pairs_train_ch[idx]一一对应的tensor方便后续训练
    return pairs_en,pairs_ch

if __name__ == '__main__':
    # import sys
    # import io
    print(len(pairs))
    print(lang1_cls.n_words)
    # print(lang1_cls.index2word)
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
    print(lang2_cls.n_words)
    # print(lang2_cls.index2word)
    print(len(pairs_test))
    print(len(pairs_train))