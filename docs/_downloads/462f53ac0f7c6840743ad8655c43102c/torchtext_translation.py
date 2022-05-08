"""
TorchText로 언어 번역하기
===================================

이 튜토리얼에서는 ``torchtext`` 를 사용하여 영어와 독일어 문장들이 포함된 잘 알려진 데이터셋을 전처리(preprocess)하고
이를 사용하여 독일어 문장을 영어로 번역하는 시퀀스-투-시퀀스(sequence-to-sequence, seq2seq) 모델을 학습하는 방법을
살펴보겠습니다.

이 튜토리얼은
PyTorch 커뮤니티 멤버인 `Ben Trevett <https://github.com/bentrevett>`__ 이 작성한
`튜토리얼 <https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>`__
에 기초하고 있으며 Ben의 허락을 받고 만들었습니다. 몇몇 기존 코드들을 제거하고 튜토리얼을 업데이트하였습니다.

이 튜토리얼을 통해 NLP 모델링을 위해 문장들을 텐서(tensor)로 전처리하고, 모델을 학습하고 검증하기 위해
`torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`__
을 사용할 수 있게 됩니다.

"""

######################################################################
# 데이터 처리하기
# --------------------------------
#
# ``torchtext`` 에는 언어 변환 모델을 만들 때 쉽게 사용할 수 있는 데이터셋을 만들기 적합한 다양한 도구가 있습니다.
# 이 예제에서는 가공되지 않은 텍스트 문장(raw text sentence)을 토큰화(tokenize)하고, 어휘집(vocabulary)을 만들고,
# 토큰을 텐서로 숫자화(numericalize)하는 방법을 알아보겠습니다.
#
# 참고 : 이 튜토리얼에서의 토큰화(tokenization)에는 `Spacy <https://spacy.io>`__ 가 필요합니다.
# Spacy는 영어 이 외의 다른 언어에 대한 강력한 토큰화 기능을 제공하기 때문에 사용합니다. ``torchtext`` 는
# `basic_english`` 토크나이저를 제공할 뿐 아니라 영어에 사용할 수 있는 다른 토크나이저들(예컨데
# `Moses <https://bitbucket.org/luismsgomes/mosestokenizer/src/default/>`__ )을 지원합니다만, 언어 번역을 위해서는 다양한 언어를
# 다루어야 하기 때문에 Spacy가 가장 적합합니다.
#
# 이 튜토리얼을 실행하려면, 우선 ``pip`` 나 ``conda`` 로 ``spacy`` 를 설치하세요. 그 다음,
# Spacy 토크나이저가 쓸 영어와 독일어에 대한 데이터를 다운로드 받습니다.
#
# ::
#
#    python -m spacy download en
#    python -m spacy download de

import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      counter.update(tokenizer(string_))
  return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

def data_process(filepaths):
  raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],
                            dtype=torch.long)
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                            dtype=torch.long)
    data.append((de_tensor_, en_tensor_))
  return data

train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)


######################################################################
# ``DataLoader``
# --------------------
# 마지막으로 사용해 볼 ``torch`` 에 특화된 기능은 바로 ``DataLoader`` 로,
# 첫 번째 인자로 데이터를 전달받기 때문에 사용하기가 쉽습니다. 문서에서도 볼 수 있듯이,
# ``DataLoader 는 데이터셋과 샘플러를 결합하고, 주어진 데이터셋에 반복 기능을 제공합니다.
# ``DataLoader`` 는 맵 형태(map-style)과 순회 가능한 형태(iterable-style) 데이터셋을 모두 지원하며,
# 단일 또는 다중 프로세스로 불러오거나, 불러오는 순서를 조정(customize)하거나
# 선택적 자동 일괄 처리(optional automatic batching), 메모리 피닝(memory pinning)을 지원합니다.
#
# 샘플 목록을 병합(merge)하여 Tensor의 미니배치를 구성하는 ``collate_fn`` (선택 사항)을 살펴보십시오.
# 맵 형태(map-style) 데이터셋을 일괄로 불러올 때 사용됩니다.

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)


######################################################################
# ``nn.Module`` 과 ``Optimizer`` 정의하기
# ------------------------------------------
# 대부분은 ``torchtext`` 가 알아서 해줍니다 : 데이터셋이 만들어지고 반복자가 정의되면, 이 튜토리얼에서
# 우리가 해야 할 일이라고는 그저 ``nn.Module`` 와 ``Optimizer`` 를 모델로서 정의하고 훈련시키는 것이 전부입니다.
#
#
# 이 튜토리얼에서 사용할 모델은 `이곳 <https://arxiv.org/abs/1409.0473>`__ 에서 설명하고 있는 구조를 따르고 있으며,
# 더 자세한 내용은 `여기 <https://github.com/SethHWeidman/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>`__
# 를 참고하시기 바랍니다.
#
# 참고 : 이 튜토리얼에서 사용하는 모델은 언어 번역을 위해 사용할 예시 모델입니다. 이 모델을 사용하는 것은
# 이 작업에 적당한 표준 모델이기 때문이지, 번역에 적합한 모델이기 때문은 아닙니다. 여러분이 최신 기술 트렌드를
# 잘 따라가고 있다면 잘 아시겠지만, 현재 번역에서 가장 뛰어난 모델은 Transformers입니다. PyTorch가
# Transformer 레이어를 구현한 내용은 `여기 <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__
# 에서 확인할 수 있으며 이 튜토리얼의 모델이 사용하는 "attention" 은 Transformer 모델에서 제안하는
# 멀티 헤드 셀프 어텐션(multi-headed self-attention) 과는 다르다는 점을 알려드립니다.


import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # 디코더로의 첫 번째 입력은 <sos> 토큰입니다.
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters())


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

######################################################################
# 참고 : 언어 번역의 성능 점수를 기록하려면, ``nn.CrossEntropyLoss`` 함수가 단순한
# 패딩을 추가하는 부분을 무시할 수 있도록 해당 색인들을 알려줘야 합니다.

PAD_IDX = en_vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

######################################################################
# 마지막으로 이 모델을 훈련하고 평가합니다 :

import math
import time


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

######################################################################
# 다음 단계
# --------------
#
# - ``torchtext`` 를 사용한 Ben Trevett의 튜토리얼을 `이곳 <https://github.com/bentrevett/>`__ 에서 확인할 수 있습니다.
# - ``nn.Transformer`` 와 ``torchtext`` 의 다른 기능들을 이용한 다음 단어 예측을 통한 언어 모델링 튜토리얼을 살펴보세요.
