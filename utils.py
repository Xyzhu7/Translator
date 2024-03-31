# -*- coding: UTF-8 -*-
import unicodedata
import re
from zhconv import convert
import random
import torch
from configs import Config

config = Config()


def unicode2Ascii(s):  # 将输入的字符串 s 转换为只包含 ASCII 字符的形式
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalizeString(s):  # 将字符全替换成小写，在标点符号前加上空格以便于分词，并修剪除英文字母和.!?之外的符号
    s = unicode2Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def cht_to_chs(line):  # 一个简易的简繁转换系统，引用自：https://github.com/gumblex/zhconv
    line = convert(line, 'zh-cn')
    line.encode('utf-8')
    return line


def get_batch_indices(total_length, batch_size):
    assert (batch_size <=
            total_length), ('Batch size is large than total data length.'
                            'Check your data or change batch size.')
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index:current_index + batch_size], current_index


def listTotensor(lang_cls, data):
    indexes = [lang_cls.word2index[word] for word in data.split(" ")]
    indexes.insert(0, config.SOS_token)
    indexes.append(config.EOS_token)
    lang_tensor = torch.tensor(indexes,
                               dtype=torch.long,
                               device=config.device)
    return lang_tensor


def tensorsFromPair(lang1_cls, lang2_cls, pair):
    lang1_tensor = listTotensor(lang1_cls, pair[0])
    lang2_tensor = listTotensor(lang2_cls, pair[1])
    return (lang1_tensor, lang2_tensor)


def idx_to_sentence(arr, vocab, insert_space=False, str=True):
    res = ''
    list = []
    for id in arr:
        word = vocab[id.item()]
        if word == 'SOS':
            continue
        elif word == 'EOS':
            break
        else:
            if insert_space:
                res += ' '
            res += word
            list.append(word)
    if str:
        return res
    else:
        return list
