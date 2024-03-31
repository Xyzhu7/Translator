# -*- coding: UTF-8 -*-
import torch
from dataset import readLangs, splitData, pairsToTensors
from model import Transformer
from configs import Config
from utils import get_batch_indices, idx_to_sentence

config = Config()
lang1 = "en"
lang2 = "cn"
path = "./sources/cmn.txt"
lang1_cls, lang2_cls, pairs = readLangs(lang1, lang2, path, config)
_, pairs_test = splitData(pairs, test_ratio=0.2, random_state=42)
pairs_test_en, pairs_test_ch = pairsToTensors(pairs_test)

def main():
    model = Transformer(src_vocab_size=lang1_cls.n_words, dst_vocab_size=lang2_cls.n_words,
                        pad_idx=config.PAD_token, d_model=config.d_model, d_ff=config.d_ff,
                        n_layers=config.n_layers, heads=config.heads,
                        dropout=config.dropout, max_seq_len=config.maxlength)

    model.to(config.device)
    model.eval()

    model_path = './model/60.pth'
    model.load_state_dict(torch.load(model_path))
    bleu = []
    for index, _ in get_batch_indices(len(pairs_test), config.batch_size):
        en_batch = pairs_test_en[index].to(config.device)
        ch_batch = pairs_test_ch[index].to(config.device)
        ch_input = torch.zeros(en_batch.shape[0], config.maxlength, dtype=torch.long).to(config.device)
        ch_input[:, 0] = 2  # <"SOS">
        with torch.no_grad():
            for i in range(1, ch_input.shape[1]):
                ch_hat = model(en_batch, ch_input)
                for j in range(en_batch.shape[0]):  # 考虑到最后一个batch取不满这里不能用batchsize
                    ch_input[j, i] = torch.argmax(ch_hat[j, i - 1])
        for i in range(en_batch.shape[0]):
            input_sentence = idx_to_sentence(en_batch[i], lang1_cls.index2word, True, True)
            output_sentence = idx_to_sentence(ch_batch[i], lang2_cls.index2word, False, True)
            output_sentence_pred = idx_to_sentence(ch_input[i], lang2_cls.index2word, False, True)
            print(input_sentence,output_sentence,output_sentence_pred)


if __name__ == '__main__':
    main()
