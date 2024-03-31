# -*- coding: UTF-8 -*-
import time
import torch
import torch.nn as nn

from configs import Config
from dataset import readLangs, splitData,pairsToTensors
from model import Transformer
from utils import get_batch_indices

config = Config()
lang1 = "en"
lang2 = "cn"
path = "./sources/cmn.txt"
lang1_cls, lang2_cls, pairs = readLangs(lang1, lang2, path, config)
pairs_train, _ = splitData(pairs, test_ratio=0.2, random_state=42)
pairs_train_en, pairs_train_ch = pairsToTensors(pairs_train)



def main():
    model = Transformer(src_vocab_size=lang1_cls.n_words, dst_vocab_size=lang2_cls.n_words,
                        pad_idx=config.PAD_token, d_model=config.d_model, d_ff=config.d_ff,
                        n_layers=config.n_layers, heads=config.heads,
                        dropout=config.dropout, max_seq_len=config.maxlength)

    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    citerion = nn.CrossEntropyLoss(ignore_index=config.PAD_token)
    tic = time.time()
    cnter = 0
    print_interval = 100

    for epoch in range(config.n_epochs):
        for index, _ in get_batch_indices(len(pairs_train), config.batch_size):
            en_batch = pairs_train_en[index].to(config.device)
            ch_batch = pairs_train_ch[index].to(config.device)
            ch_input = ch_batch[:, :-1]
            ch_label = ch_batch[:, 1:]
            ch_hat = model(en_batch, ch_input)

            ch_label_mask = (ch_label != config.PAD_token)
            preds = torch.argmax(ch_hat, -1)
            correct = (preds == ch_label)
            acc = torch.sum(ch_label_mask * correct) / torch.sum(ch_label_mask)

            n, seq_len = ch_label.shape
            ch_hat = torch.reshape(ch_hat, (n * seq_len, -1))
            ch_label = torch.reshape(ch_label, (n * seq_len,))
            loss = citerion(ch_hat, ch_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if cnter % print_interval == 0:
                toc = time.time()
                interval = toc - tic
                seconds = int(interval % 60)
                minutes = int((interval // 60) % 60)
                hours = int(interval // 3600)
                print(f'{epoch:02d} {cnter:08d} {hours:02d}:{minutes:02d}:{seconds:02d}'
                      f' loss: {loss.item()} acc: {acc.item()}')
            cnter += 1

        if (epoch + 1) % 5 == 0:
            model_path = f"./model/{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
