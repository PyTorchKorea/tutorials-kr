# -*- coding: utf-8 -*-
"""
(베타) LSTM 기반 단어 단위 언어 모델의 동적 양자화
==================================================================

**Author**: `James Reed <https://github.com/jamesr66a>`_

**Edited by**: `Seth Weidman <https://github.com/SethHWeidman/>`_

**번역**: `Myungha Kwon <https://github.com/kwonmha/>`_

시작하기
------------

양자화는 모델의 크기를 줄이고 추론 속도를 높이면서도 정확도는 별로 낮아지지 않도록,
모델의 가중치와 활성 함수를 실수형에서 정수형으로 변환합니다.

이 튜토리얼에서, PyTorch의 `단어 단위 언어 모델 <https://github.com/pytorch/examples/tree/master/word_language_model>`_
예제를 따라하면서, LSTM 기반의 단어 예측 모델에 가장 간단한 양자화 기법인
`동적 양자화 <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_
를 적용해 보겠습니다.
"""

# imports
import os
from io import open
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################
# 1. 모델 정의하기
# -------------------
#
# 여기서, 단어 단위 언어 모델 예제에서 사용된 `모델 <https://github.com/pytorch/examples/blob/master/word_language_model/model.py>`_ 을
# 따라 LSTM 모델을 정의합니다.

class LSTMModel(nn.Module):
    """인코더, 순환 모듈, 디코더를 갖고 있는 모듈"""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

######################################################################
# 2. 텍스트 데이터 불러오기
# --------------------------
#
# 다음으로, 단어 단위 언어 모델 예제의 `전처리 과정 <https://github.com/pytorch/examples/blob/master/word_language_model/data.py>`_ 을
# 따라 `Wikitext-2 데이터셋 <https://www.google.com/search?q=wikitext+2+data>`_ 을
# `Corpus` 인스턴스에 불러옵니다.

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """텍스트 파일을 토큰화 합니다."""
        assert os.path.exists(path)
        # 사전에 단어를 추가합니다.
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # 파일 내용을 토큰화 합니다.
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

model_data_filepath = 'data/'

corpus = Corpus(model_data_filepath + 'wikitext-2')

######################################################################
# 3. 사전학습된 모델 불러오기
# -----------------------------
#
# 이 튜토리얼은 학습된 모델에 적용하는 양자화 기법인 동적 양자화에 대한
# 튜토리얼입니다. 그래서 단순히, 사전학습된 가중치들을 모델에 불러올
# 것입니다. 이 가중치들은 단어 단위 언어 모델 예제에서 기본 설정으로
# 5 에폭만큼 학습하여 얻었습니다.

ntokens = len(corpus.dictionary)

model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
        )
    )

model.eval()
print(model)

######################################################################
# 이제 사전학습된 모델이 잘 동작하는지 확인해보기 위해 텍스트를 생성해
# 보겠습니다. 지금까지 튜토리얼을 진행했던 방식처럼 `이 예제 <https://github.com/pytorch/examples/blob/master/word_language_model/generate.py>`_ 를
# 따라 하겠습니다.

input_ = torch.randint(ntokens, (1, 1), dtype=torch.long)
hidden = model.init_hidden(1)
temperature = 1.0
num_words = 1000

with open(model_data_filepath + 'out.txt', 'w') as outf:
    with torch.no_grad():  # 기록을 추적하지 않습니다.
        for i in range(num_words):
            output, hidden = model(input_, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(str(word.encode('utf-8')) + ('\n' if i % 20 == 19 else ' '))

            if i % 100 == 0:
                print('| Generated {}/{} words'.format(i, 1000))

with open(model_data_filepath + 'out.txt', 'r') as outf:
    all_output = outf.read()
    print(all_output)

######################################################################
# 이 모델이 GPT-2는 아니지만, 언어의 구조를 배우기 시작한 것처럼 보입니다!.
#
# 동적 양자화를 설명하기 위한 준비가 거의 다 됐습니다. 몇 개의 보조 함수만 정의하면 됩니다.

bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# 시험용 데이터 셋을 생성합니다.
def batchify(data, bsz):
    # 얼마나 깔끔하게 데이터셋을 bsz 개의 부분으로 나눌 수 있는지 계산합니다.
    nbatch = data.size(0) // bsz
    # 데이터에서 쓰이지 않는 잉여 부분들을 제거합니다.
    data = data.narrow(0, 0, nbatch * bsz)
    # 데이터를 bsz로 등분합니다.
    return data.view(bsz, -1).t().contiguous()

test_data = batchify(corpus.test, eval_batch_size)

# 평가 함수들
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
  """은닉 상태를 변화도 기록에서 제거된 새로운 tensor로 만듭니다."""

  if isinstance(h, torch.Tensor):
      return h.detach()
  else:
      return tuple(repackage_hidden(v) for v in h)

def evaluate(model_, data_source):
    # Dropout을 중지시키는 평가 모드로 실행합니다.
    model_.eval()
    total_loss = 0.
    hidden = model_.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model_(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

######################################################################
# 4. 동적 양자화 테스트하기
# ----------------------------
#
# 마침내, 모델에 ``torch.quantization.quantize_dynamic`` 을 호출할 수
# 있게 됐습니다. 구체적으로,
#
# - 모델의 ``nn.LSTM`` 과 ``nn.Linear`` 모듈을 양자화 하도록 명시합니다.
# - 가중치들이 ``int8`` 형태로 변환되도록 명시합니다.

import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)

######################################################################
# 모델에 변화가 없어 보입니다. 양자화는 어떤 효과가 있었을까요? 우선,
# 모델의 크기가 많이 줄었습니다.

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)

######################################################################
# 두 번째로, 평가 손실값은 같으나 추론 속도가 빨라졌습니다.

# 메모: 단일 쓰레드 환경에서 비교하기 위해 쓰레드의 수를 1로 설정했습니다. 양자화된
# 모델은 단일 쓰레드에서 실행되기 때문입니다.

torch.set_num_threads(1)

def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)
    elapsed = time.time() - s
    print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

time_model_evaluation(model, test_data)
time_model_evaluation(quantized_model, test_data)

######################################################################
# 이 모델을 맥북 프로에서 양자화 하지 않고 실행하면 추론하는데 약 200초가 소요되고,
# 양자화 하면 약 100초만 소요됩니다.
#
# 마치며
# ----------
#
# 동적 양자화는 정확도를 거의 낮추지 않으면서도 모델의 크기를 줄일 수 있는
# 간편한 방법이 될 수 있습니다.
#
# 읽어주셔서 감사합니다. 언제나처럼 어떠한 피드백도 환영이니, 의견이 있다면
# `여기 <https://github.com/pytorch/pytorch/issues>`_ 에 이슈를 남겨 주세요.
