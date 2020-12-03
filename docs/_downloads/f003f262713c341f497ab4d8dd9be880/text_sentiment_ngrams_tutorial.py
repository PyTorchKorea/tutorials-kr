"""
TorchText로 텍스트 분류하기
==================================
**번역**: `김강민 <https://github.com/gangsss>`_ , `김진현 <https://github.com/lewhe0>`_

이 튜토리얼에서는 ``torchtext`` 에 포함되어 있는 텍스트 분류
데이터셋의 사용 방법을 살펴 봅니다. 데이터셋은 다음을 포함합니다.

::

   - AG_NEWS,
   - SogouNews,
   - DBpedia,
   - YelpReviewPolarity,
   - YelpReviewFull,
   - YahooAnswers,
   - AmazonReviewPolarity,
   - AmazonReviewFull

이 예제에서는 ``TextClassification`` 의 데이터셋들 중 하나를 이용해 분류를 위한
 지도 학습 알고리즘을 훈련하는 방법을 보여줍니다.

ngrams를 이용하여 데이터 불러오기
-----------------------------------

Bag of ngrams 피쳐는 지역(local) 단어 순서에 대한 부분적인 정보를 포착하기 위해 적용합니다.
실제 상황에서는 bi-gram이나 tri-gram은 단 하나의 단어를 이용하는 것보다 더 많은 이익을 주기 때문에 적용됩니다.
예를 들면 다음과 같습니다.

::

   "load data with ngrams"
   Bi-grams 결과: "load data", "data with", "with ngrams"
   Tri-grams 결과: "load data with", "data with ngrams"

``TextClassification`` 데이터셋은 ngrams method을 지원합니다. ngrams을 2로 설정하면,
데이터셋 안의 예제 텍스트는 각각의(single) 단어들에 bi-grams 문자열이 더해진 리스트가 될 것입니다.

"""

import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os
if not os.path.isdir('./.data'):
	os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



######################################################################
# 모델 정의하기
# -------------
#
# 우리의 모델은
# `EmbeddingBag <https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag>`__
# 레이어와 선형 레이어로 구성됩니다 (아래 그림 참고).
# ``nn.EmbeddingBag``는 임베딩들로 구성된 '가방'의 평균을 계산합니다.
# 이때 텍스트(text)의 각 원소는 그 길이가 다를 수 있습니다. 텍스트의
# 길이는 오프셋(offset)에 저장되어 있으므로 여기서 ``nn.EmbeddingBag``
# 에 패딩을 사용할 필요는 없습니다.
#
# 덧붙여서, ``nn.EmbeddingBag`` 은 임베딩의 평균을 즉시 계산하기 때문에,
# 텐서들의 시퀀스를 처리할 때 성능 및 메모리 효율성 측면에서의 장점도
# 갖고 있습니다.
#
# .. image:: ../_static/img/text_sentiment_ngrams_model.png
#


import torch.nn as nn
import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


######################################################################
# 인스턴스 생성하기
# -----------------
#
# AG_NEWS 데이터셋에는 4 종류의 레이블이 달려 있으며, 따라서 클래스의 개수도 4개 입니다.
#
# ::
#
#    1 : World (세계)
#    2 : Sports (스포츠)
#    3 : Business (경제)
#    4 : Sci/Tec (과학/기술)
#
# 어휘집의 크기(Vocab size)는 어휘집(vocab)의 길이와 같습니다 (여기에는
# 각각의 단어와 ngrame이 모두 포함됩니다). 클래스의 개수는 레이블의 종류
# 수와 같으며, AG_NEWS의 경우에는 4개 입니다.
#

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)


######################################################################
# 배치 생성을 위한 함수들
# -----------------------
#


######################################################################
# 텍스트 원소의 길이가 다를 수 있으므로, 데이터 배치와 오프셋을 생성하기
# 위한 사용자 함수 generate_batch()를 사용하려 합니다. 이 함수는
# ``torch.utils.data.DataLoader`` 의 ``collate_fn`` 인자로 넘겨줍니다.
#
# ``collate_fn`` 의 입력은 그 크기가 batch_size인 텐서들의 리스트이며,
# ``collate_fn`` 은 이들을 미니배치로 묶는 역할을 합니다. 여러분이
# 주의해야 할 점은, ``collate_fn`` 를 선언할 때 최상위 레벨에서 정의해야
# 한다는 점입니다. 그래야 이 함수를 각각의 워커에서 사용할 수 있음이
# 보장됩니다.
#
# 원본 데이터 배치 입력의 텍스트 원소들은 리스트 형태이며, 이들을 하나의
# 텐서가 되도록 이어 붙인 것이 ``nn.EmbeddingBag`` 의 입력이 됩니다.
# 오프셋은 텍스트의 경계를 나타내는 텐서이며, 각 원소가 텍스트 텐서의
# 어느 인덱스에서 시작하는지를 나타냅니다. 레이블은 각 텍스트 원소의
# 레이블을 담고 있는 텐서입니다.
#

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum은 dim 차원의 요소들의 누적 합계를 반환합니다.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


######################################################################
# 모델을 학습하고 결과를 평가하는 함수 정의하기
# ---------------------------------------------
#


######################################################################
# PyTorch 사용자라면
# `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`__
# 를 활용하는 것을 추천합니다. 또한 이를 사용하면 데이터를 쉽게 병렬적으로
# 읽어올 수 있습니다 (이에 대한 튜토리얼은 `이 문서 <https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html>`__
# 를 참고하시기 바랍니다). 우리는 여기서 ``DataLoader`` 를 이용하여
# AG_NEWS 데이터셋을 읽어오고, 이를 모델로 넘겨 학습과 검증을 진행합니다.
#

from torch.utils.data import DataLoader

def train_func(sub_train_):

    # Train the model
    # 모델을 학습합니다
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # 학습률을 조절합니다
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


######################################################################
# 데이터셋을 분할하고 모델 수행하기
# ---------------------------------
#
# 원본 AG_NEWS에는 검증용 데이터가 포함되어 있지 않기 때문에, 우리는 학습
# 데이터를 학습 및 검증 데이터로 분할하려 합니다. 이때 데이터를 분할하는
# 비율은 0.95(학습)와 0.05(검증) 입니다. 우리는 여기서 PyTorch의
# 핵심 라이브러리 중 하나인
# `torch.utils.data.dataset.random_split <https://pytorch.org/docs/stable/data.html?highlight=random_split#torch.utils.data.random_split>`__
# 함수를 사용합니다.
#
# `CrossEntropyLoss <https://pytorch.org/docs/stable/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__
# 기준(criterion)은 각 클래스에 대해 nn.LogSoftmax()와 nn.NLLLoss()를
# 합쳐 놓은 방식입니다.
# `SGD <https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html>`__
# optimizer는 확률적 경사 하강법를 구현해놓은 것입니다. 처음의 학습률은
# 4.0으로 두었습니다. 매 에폭을 진행하면서 학습률을 조절할 때는
# `StepLR <https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#StepLR>`__
# 을 사용합니다.
#

import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


######################################################################
# 이 모델을 GPU 상에서 수행했을 때 다음과 같은 결과를 얻었습니다.
#
# Epoch: 1 \| time in 0 minutes, 11 seconds (에폭 1, 수행 시간 0분 11초)
#
# ::
#
#        Loss: 0.0263(train)     |       Acc: 84.5%(train)
#        Loss: 0.0001(valid)     |       Acc: 89.0%(valid)
#
#
# Epoch: 2 \| time in 0 minutes, 10 seconds (에폭 2, 수행 시간 0분 10초)
#
# ::
#
#        Loss: 0.0119(train)     |       Acc: 93.6%(train)
#        Loss: 0.0000(valid)     |       Acc: 89.6%(valid)
#
#
# Epoch: 3 \| time in 0 minutes, 9 seconds (에폭 3, 수행 시간 0분 9초)
#
# ::
#
#        Loss: 0.0069(train)     |       Acc: 96.4%(train)
#        Loss: 0.0000(valid)     |       Acc: 90.5%(valid)
#
#
# Epoch: 4 \| time in 0 minutes, 11 seconds (에폭 4, 수행 시간 0분 11초)
#
# ::
#
#        Loss: 0.0038(train)     |       Acc: 98.2%(train)
#        Loss: 0.0000(valid)     |       Acc: 90.4%(valid)
#
#
# Epoch: 5 \| time in 0 minutes, 11 seconds (에폭 5, 수행 시간 0분 11초)
#
# ::
#
#        Loss: 0.0022(train)     |       Acc: 99.0%(train)
#        Loss: 0.0000(valid)     |       Acc: 91.0%(valid)
#


######################################################################
# 평가 데이터로 모델 평가하기
# ---------------------------
#

print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')


######################################################################
# 평가 데이터셋을 통한 결과를 확인합니다...

#
# ::
#
#        Loss: 0.0237(test)      |       Acc: 90.5%(test)
#


######################################################################
# 임의의 뉴스로 평가하기
# ----------------------
#
# 현재까지 구한 최고의 모델로 골프 뉴스를 테스트해보려 합니다. 레이블에
# 대한 정보는
# `여기에 <https://pytorch.org/text/datasets.html?highlight=ag_news#torchtext.datasets.AG_NEWS>`__
# 나와 있습니다.
#

import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])

######################################################################
# This is a Sports news (스포츠 뉴스)
#


######################################################################
# 이 튜토리얼에서 사용한 예제 코드는
# `여기에서 <https://github.com/pytorch/text/tree/master/examples/text_classification>`__
# 확인하실 수 있습니다.
