"""
TorchText을 활용한 텍스트 분류
==================================

이 튜토리얼은 ``torchtext`` 안의 텍스트 분류 데이터셋을 어떻게 이용하는 지를 보여줍니다.
데이터셋에는
::

   - AG_NEWS,
   - SogouNews,
   - DBpedia,
   - YelpReviewPolarity,
   - YelpReviewFull,
   - YahooAnswers,
   - AmazonReviewPolarity,
   - AmazonReviewFull
이 포함되어 있습니다.

이 예제에서는 ``TextClassification`` 의 데이터셋들 중 하나를 이용해 분류를 위한
 지도 학습 알고리즘을 훈련하는 방법을 보여줍니다.

데이터를 ngrams과 함께 불러옵니다.
---------------------


bag of ngrams 피쳐는 지역(local) 단어 순서에 대한 부분적인 정보를 포착하기 위해 적용합니다.
실제는 bi-gram이나 tri-gram은 단 하나의 단어를 이용하는 것보다 더 많은 이익을 주기 때문에 적용됩니다.
예를 들어,


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
# 모델을 정의합니다.
# ----------------
#
# 모델은  
# `EmbeddingBag <https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag>`__
# 계층과 선형 계층으로 이뤄져있습니다. (아래 그림을 참고하세요). ``nn.EmbeddingBag``
# 은 임베딩 bag의 평균 값을 계산합니다. 텍스트 항목들은 서로 다른 길이를 갖고 있습니다.
# ``nn.EmbeddingBag`` 텍스트의 길이가 offsets에 저장되기 때문에 패딩이 필요하지 않습니다.
#
#
# 추가적으로, ``nn.EmbeddingBag`` 은 상황에 따라 임베딩 값들에 대한 평균을 축적하기 때문에, 
# ``nn.EmbeddingBag`` 은 텐서들의 시퀀스를 처리하기 위한 메모리 효율상과 성능을 향상 시킬 수 있습니다.
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

# 인스턴스 시작
# --------------------
#
# AG_NEWS 데이터셋은 4개의 라벨이 있고 그에 따른 클래스 수도 4개입니다.
#
# ::
#
#    1 : World
#    2 : Sports
#    3 : Business
#    4 : Sci/Tec
#
# 단어의 사이즈는 단어의 길이와 같습니다.(각 단어와 n-grams을 포함하여)
# 클래스의 수는 라벨의 수와 같습니다. AG_NEWS의 경우는 4개입니다.
#

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)


######################################################################
# 배치 생성에 사용되는 함수
# --------------------------------
#


######################################################################
# 텍스트 항목의 길이가 다르기 때문에, 사용자 정의 함수 generate_batch()를 사용하여 데이터 배치와 오프셋을 생성합니다.
# 이 함수는 ``torch.utils.data.DataLoader`` 안의 ``collate_fn`` 를 통과합니다.
# 
#  ``collate_fn`` 에 대한 입력은 batch_size 크기를 갖는 tensors의 리스트이고,
#  ``collate_fn`` 함수는 이들을 mini-batch로  둡니다. ``collate_fn`` 이 가장  높은 level에서 선언되었다는 것을 기억해야합니다.
# 이는 함수가 각 worker마다 이용가능하게 하는 것을 보장해줍니다.
#
# 원본 데이터 배치 인풋 안의 텍스트들은 리스트에 들어가고 단일 텐서로서 ``nn.EmbeddingBag`` 의 입력으로 연결합니다.
# offsets은 텍스트 텐서에서 개별 시퀀스의 시작 인덱스를 표현하는 구분자들의 텐서입니다.
# label은 각 텍스트 항목별 라벨을 저장해 두는 tensor입니다.
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
# 모델을 훈련하고 결과를 평가하는 함수를 정의
# ---------------------------------------------------------
#


######################################################################
#
# `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`__
# 은 데이터을 병렬적으로 로딩할 수있어 Pytorch 유저들에게 추천합니다.
# (튜토리얼은
# `여기서 <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`__).
# 우리는 여기서 ``DataLoader`` 를  AG_NEWS 데이터셋을 로드하고 훈련과 검증을 위한
# 모델링에 보냅니다.
#

from torch.utils.data import DataLoader

def train_func(sub_train_):

    # Train the model
    # 모델을 훈련합니다
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

    # Adjust the learning rate
    # learning rate를 조정합니다.
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
# 데이터 셋을 나누고 모델을 실행합니다.
# -----------------------------------
#
# 원래 AG_NEWS에는 검증 데이터셋이 없기 때문에, 우리는 훈련데이터셋을 0.95(훈련데이터셋)과 
# 0.05(검증데이터셋)인 비율로 나눈다.  
# 여기서 우리는 PyTorch core library 안에 있는 `torch.utils.data.dataset.random_split <https://pytorch.org/docs/stable/data.html?highlight=random_split#torch.utils.data.random_split>`__
# 함수를 사용합니다.
#
# `CrossEntropyLoss <https://pytorch.org/docs/stable/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__
# 기준은 단일 클래스안에서 nn.LogSoftmax() 과 nn.NLLLoss()을 결합합니다.
# 이는 C개의 클래스들을 분류하는 문제에서 유용합니다.
# `SGD <https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html>`__
# 는 옵티마이저로써 stochastic gradient descent method으로 적용됩니다. 시작 러닝레이트는 4.0으로 설정합니다.
# `StepLR <https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#StepLR>`__
# 은 여기서 러닝레이트를 에포크에 따라 조정하기 위해 사용됩니다.
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
# GPU상에서 모델을 실행시킵니다.
#
# Epoch: 1 \| time in 0 minutes, 11 seconds
#
# ::
#
#        Loss: 0.0263(train)     |       Acc: 84.5%(train)
#        Loss: 0.0001(valid)     |       Acc: 89.0%(valid)
#
#
# Epoch: 2 \| time in 0 minutes, 10 seconds
#
# ::
#
#        Loss: 0.0119(train)     |       Acc: 93.6%(train)
#        Loss: 0.0000(valid)     |       Acc: 89.6%(valid)
#
#
# Epoch: 3 \| time in 0 minutes, 9 seconds
#
# ::
#
#        Loss: 0.0069(train)     |       Acc: 96.4%(train)
#        Loss: 0.0000(valid)     |       Acc: 90.5%(valid)
#
#
# Epoch: 4 \| time in 0 minutes, 11 seconds
#
# ::
#
#        Loss: 0.0038(train)     |       Acc: 98.2%(train)
#        Loss: 0.0000(valid)     |       Acc: 90.4%(valid)
#
#
# Epoch: 5 \| time in 0 minutes, 11 seconds
#
# ::
#
#        Loss: 0.0022(train)     |       Acc: 99.0%(train)
#        Loss: 0.0000(valid)     |       Acc: 91.0%(valid)
#


######################################################################
# 테스트 데이터 셋을 통해 모델을 평가합니다.
# ------------------------------------
#

print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')


######################################################################
# 테스트 데이터셋을 통한 결과를 확인합니다...
#
# ::
#
#        Loss: 0.0237(test)      |       Acc: 90.5%(test)
#


######################################################################
# 랜덤 뉴스들을 통해 실험합니다.
# ---------------------
#
# 가장 좋았던 모델을 사용하여 골프 뉴스를 테스트 해보세요. 라벨에 대한 정보는 밑에 나와있습니다.
# `here <https://pytorch.org/text/datasets.html?highlight=ag_news#torchtext.datasets.AG_NEWS>`__

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
# 이것은 스포츠 뉴스입니다,


######################################################################
# 코드의 결과물들은 밑의 노트에서 찾을 수 있습니다.
#
# `here <https://github.com/pytorch/text/tree/master/examples/text_classification>`__
