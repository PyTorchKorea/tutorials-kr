"""
(베타) LSTM 언어 모델에서의 동적 양자화
==================================================================

**Author**: `James Reed <https://github.com/jamesr66a>`_

**Edited by**: `Seth Weidman <https://github.com/SethHWeidman/>`_

**번역**: `박경림 <https://github.com/kypark7/>`_

개요
------------

양자화는 모델의 가중치와 활성화를 float에서 int로 변환하는 작업이 포함되며,
적중률에 주는 타격을 최소화하면서 모델 크기를 줄이고 추론(inference) 속도를 높일 수 있습니다.

이 튜토리얼에서는 PyTorch 예제의
`word language 모델 <https://github.com/pytorch/examples/tree/master/word_language_model>`_ 을 따라
LSTM 기반 next word-prediction 모델에 가장 쉬운 형태인 -
`동적 양자화 <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_ -
를 적용합니다.
"""

# 불러오기
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
# 여기서 우리는 word language 모델 예제의
# `모델 <https://github.com/pytorch/examples/blob/master/word_language_model/model.py>`_
# 을 따라 LSTM 모델 아키텍처를 정의합니다.

class LSTMModel(nn.Module):
    """인코더, 반복 모듈 및 디코더가 있는 컨테이너 모듈."""

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
# 2. 텍스트 데이터에 로드
# ------------------------
#
# 다음으로, word language 모델 예제의 
# `전처리 <https://github.com/pytorch/examples/blob/master/word_language_model/data.py>`_
# 에 따라
# `Wikitext-2 데이터셋 <https://www.google.com/search?q=wikitext+2+data>`_
# 을 `Corpus` 에 다시 로드합니다.

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
        """텍스트 파일 토큰화"""
        assert os.path.exists(path)
        # 사전에 단어 추가
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # 파일 내용 토큰화
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
# 3. 사전 훈련된 모델 불러오기
# -----------------------------
#
# 이 튜토리얼은 모델이 학습된 후 적용되는 양자화 기술인 동적 양자화에 대한 튜토리얼입니다.
# 따라서 우리는 미리 학습된 가중치를 모델 아키텍처에 
# 로드할 것 입니다; 이 가중치는 word language 모델 예제의 기본 설정을
# 사용하여 5개의 epoch 동안 학습하여 얻은 것입니다.
# 

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
# 이제 사전 학습된 모델이 제대로 작동하는지 확인하기 위해 몇 가지 텍스트를
# 생성하겠습니다 - 이전과 비슷하게, 우리는 
# `여기 <https://github.com/pytorch/examples/blob/master/word_language_model/generate.py>`_ 를 따릅니다.

input_ = torch.randint(ntokens, (1, 1), dtype=torch.long)
hidden = model.init_hidden(1)
temperature = 1.0
num_words = 1000

with open(model_data_filepath + 'out.txt', 'w') as outf:
    with torch.no_grad():  # no tracking history
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
# GPT-2는 아니지만 모델이 언어 구조를 배우기 시작한 것으로
# 보여집니다!
#
# 동적 양자화를 시연할 준비가 거의 끝났습니다. 몇 가지 helper 함수를 정의하기만
# 하면 됩니다:

bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# 테스트 데이터셋 만들기
def batchify(data, bsz):
    # 데이터셋을 bsz 부분으로 얼마나 깔끔하게 나눌 수 있는지 계산해봅니다.
    nbatch = data.size(0) // bsz
    # 깔끔하게 맞지 않는 추가적인 부분(나머지들)을 잘라냅니다.
    data = data.narrow(0, 0, nbatch * bsz)
    # 데이터에 대하여 bsz 배치들로 동등하게 나눕니다.
    return data.view(bsz, -1).t().contiguous()

test_data = batchify(corpus.test, eval_batch_size)

# 평가 기능
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
  """hidden states를 새 텐서로 래핑하여 기록에서 분리"""

  if isinstance(h, torch.Tensor):
      return h.detach()
  else:
      return tuple(repackage_hidden(v) for v in h)

def evaluate(model_, data_source):
    # 드랍아웃을 비활성화하는 평가모드 키기
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
# 4. 동적 양자화 테스트
# ----------------------------
#
# 마지막으로 모델에서 ``torch.quantization.quantize_dynamic`` 을 호출 할 수 있습니다!
# 구체적으로,
#
# - 모델의 ``nn.LSTM`` 그리고 ``nn.Linear`` 모듈이 양자화되도록
#   지정합니다.
# - 가중치가 ``int8`` 값으로 변환하도록 지정합니다.

import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)

######################################################################
# 모델은 동일하게 보입니다; 이것이 어떤 이득을 주는 것일까요? 첫째, 모델 크기가
# 상당히 줄어 듭니다:

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)

######################################################################
# 둘째, evaluation loss의 차이없이 더 빠른 추론(inference) 시간을 볼 수 있습니다:
#
# Note: 양자화 된 모델은 단일 스레드로 실행되기 때문에 단일 스레드 비교를 위해
# 스레드 수를 하나로 만듭니다.

torch.set_num_threads(1)

def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)
    elapsed = time.time() - s
    print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

time_model_evaluation(model, test_data)
time_model_evaluation(quantized_model, test_data)

######################################################################
# MacBook Pro에서 로컬로 실행하는 경우, 양자화 없이는 추론(inference)에 약 200초가 걸리고 
# 양자화를 사용하면 약 100초가 걸립니다.
#
# 결론
# ----------
#
# 동적 양자화는 정확도에 제한적인 영향을 미치면서 모델 크기를 줄이는
# 쉬운 방법이 될 수 있습니다.
#
# 읽어주셔서 감사합니다! 항상 그렇듯이, 우리는 피드백을 환영하므로 문제가 있으면
# `여기 <https://github.com/pytorch/pytorch/issues>`_ 에 이슈를 만들어주세요.
