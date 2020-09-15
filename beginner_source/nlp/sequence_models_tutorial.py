# -*- coding: utf-8 -*-
r"""
시퀀스 모델과 LSTM(Long-Short Teram Memory)
===================================================

**저자**: `Robert Guthrie`
**번역**: `정신유 <https://github.com/SSinyu>`

지금까지 다양한 피드 포워드(feed-forward) 네트워크에 대해 알아 보았습니다.
이러한 네트워크는 유지되는 상태(state)가 전혀 없으며, 이는 우리가 원하는 것이
아닐 수 있습니다. 시퀀스 모델은 자연어 처리(NLP)의 핵심이며, 이러한 모델은
입력 데이터들 사이에 시간적인 의존성이 있는 모델입니다. 시퀀스 모델의 예로는 
품사(part-of-speech) 태깅에 사용되는 Hidden Markove Model 및 Conditional 
random field가 있습니다.

RNN(Recurrent Neural Network)은 어떠한 상태(state)를 유지하는 네트워크 
입니다. 예를 들면, RNN의 출력은 다음 입력의 일부로 사용되어 네트워크가 
시퀀스를 통과하며 정보가 전파될 수 있습니다. LSTM의 경우 시퀀스의 각 원소에
해당하는 *은닉 상태(hidden state)*인 :math:`h_t` 가 있으며, 이는 원칙적으로
시퀀스의 이전 임의 지점의 정보를 포함할 수 있습니다. Hidden state를 사용해 
언어 모델(language model), 품사 태그 및 다른 많은 단어를 예측할 수 있습니다.


Pytorch를 이용한 LSTM
~~~~~~~~~~~~~~~~~

LSTM 예제를 시작하기 전에 몇 가지 유의사항이 있습니다. Pytorch LSTM의 모든 
입력은 3D 텐서이며 텐서 내 각각의 축의 의미는 중요합니다. 첫번째 축은 시퀀스 
자체이며, 두번째 축은 미니 배치 내의 각 데이터의 인덱스, 세번째 축은 입력 
원소의 인덱스 입니다. 여기서는 미니 배치에 대해 논의하지 않으므로 두번째 축에 
대해 무시하고 항상 1차원만 있다고 가정하겠습니다. 만약 "The cow jumped" 라는 
문장에 대해 시퀀스 모델을 이용하려 할 때, 입력은 다음과 같아야 합니다.

.. math::


   \begin{bmatrix}
   \overbrace{q_\text{The}}^\text{row vector} \\
   q_\text{cow} \\
   q_\text{jumped}
   \end{bmatrix}

크기가 1인 두번째 추가 차원이 있음을 기억하세요.

또한 첫번째 축도 크기가 1인 경우, 한 번에 하나의 시퀀스를 통과시킬 수 있습니다.

간단한 예제를 살펴봅시다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

######################################################################

lstm = nn.LSTM(3, 3)  # 입력 및 출력 차원은 모두 3차원입니다.
inputs = [torch.randn(1, 3) for _ in range(5)]  # 길이 5의 시퀀스를 만듭니다.

# 은닉 상태(hidden state)를 초기화 합니다.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # 한 번에 시퀀스 하나의 원소로 진행합니다.
    # 각 단계 후에, 아래의 "hidden"은 은닉 상태(hidden state)를 포함합니다.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# 대안으로 전체 시퀀스를 한번에 수행 할 수 있습니다.
# LSTM이 반환하는 첫 번째 값은 시퀀스 전체에 대한 모든 은닉 상태(hidden state)입니다.
# 두 번째 값은 가장 최근의 은닉 상태(hidden state)입니다.
# (위에서 "hidden"의 마지막 텐서와 "out"을 비교해보면 같음을 확인할 수 있습니다.)
# 그 이유는 다음과 같습니다.
# "out"은 시퀀스 내의 모든 은닉 상태(hidden state)에 접근합니다.
# "hidden"은 이후 lstm의 인자(argument)로 전달되어 시퀀스를 계속하고 역전파 할 수 
# 있습니다.
# 추가로 두번째 차원을 더합니다.
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)


######################################################################
# 예제: 형태소 분석(Part-of-Speech tagging)을 위한 LSTM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이번에는 LSTM을 이용해 형태소 분석을 진행합니다. 이번 예제에서 우리는 
# Viterbi 알고리즘이나 Forward-Backward 알고리즘 또는 이와 유사한 방법들을 
# 사용하지 않습니다. 하지만 이 글을 읽는 분들에게 (도전적인) 과제로써, 
# 이번 예제에서 어떠한 일이 일어나고 있는지 확인한 후 Viterbi 알고리즘을 어떻게 
# 사용할 수 있는지 생각해 보시기 바랍니다.
#
# 모델은 다음과 같습니다.
# 입력 문장을 :math:`w_1, \dots, w_M` 라고 합시다. 여기서 :math:`w_i \in V`
# 은 단어입니다. 또한 :math:`T`를 태그 셋으로, :math:`y_i`를 단어 :math:`w_i`
# 의 태그라고 합시다. :math:`\hat{y}_i`에 의해 단어 :math:`w_i`의 태그에 대한
# 예측을 나타냅니다.
#
# 이것은 구조 예측 모델이며, 출력은 :math:`\hat{y}_i \in T`인
# :math:`\hat{y}_1, \dots, \hat{y}_M` 시퀀스입니다. 
#
# 예측을 수행하려면 문장을 LSTM에 전달해야 합니다. 타임 스텝 :math:`i`의
# 은닉 상태(hidden state)를 :math:`h_i`로 표시합니다. 또한 각 태그에 고유한
# 인덱스를 할당해야 합니다 (단어 임베딩 섹션에서 word\_to\_ix 가 있는 것과 
# 같습니다). 그러면 :math:`\hat{y}_i`에 대한 예측은 아래와 같습니다.
#
# .. math::  \hat{y}_i = \text{argmax}_j \  (\log \text{Softmax}(Ah_i + b))_j
#
# 즉, 은닉 상태(hidden state)의 affine map에 log softmax를 취하며, 이 벡터
# 에서 최대값을 갖는 태그가 예측된 태그입니다. 이는 :math:`A`의 targer space의
# 차원이 :math:`|T|`임을 의미합니다.
#
#
# 데이터 준비하기:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# 아래와 같이 차원을 작게하면 학습 과정에서 가중치가 어떻게 변하는 지 
# 볼 수 있습니다. 이들은 일반적으로 32 또는 64 차원 이상으로 사용됩니다.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

######################################################################
# 모델 생성하기:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM은 단어 임베딩을 입력으로 하며, hidden_dim 차원의
        # 은닉 상태(hidden state)를 출력합니다. 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # 은닉 상태 공간(hidden state space)에서 태그 공간(tag space)로 
        # 매핑하는 선형 계층입니다.
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# 모델 학습하기:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습하기 전에 스코어를 확인합니다.
# 출력의 요소 i,j는 단어 i에 해당하는 태그 j에 대한 스코어입니다.
# 여기에서는 학습할 필요가 없으므로 torch.no_grad()을 이용합니다.
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  
    for sentence, tags in training_data:
        # 단계 1. Pytorch가 그래디언트를 축적합니다.
        model.zero_grad()

        # 단계 2. 네트워크에 대한 입력을 준비합니다. 
        # 즉, 단어 인덱스의 텐서로 변환합니다.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # 단계 3. 순방향 전파(forward pass)를 실행합니다.
        tag_scores = model(sentence_in)

        # 단계 4. optimizer.step()을 호출하여 손실과 그래디언트를 계산하고
        # 파라미터를 업데이트 합니다.
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# 학습 후 스코어 확인하기
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # 문장은 "the dog ate the apple" 입니다. i,j는 단어 i에 대한 태그 j의
    # 스코어에 해당합니다. 예측된 태그는 최대 스코어를 갖는 태그 입니다.
    # 여기서 0은 1행의 최대값에 해당하는 인덱스이며, 1은 2행의 최대값에 해당하는
    # 인덱스 등 이므로 예측된 시퀀스가 아래와 같이 0 1 2 0 1 임을 볼 수 있습니다. 
    # 이 숫자는 DET NOUN VERB DET NOUN 시퀀스가 맞습니다.
    print(tag_scores)


######################################################################
# 연습: 문자 수준(character-level) 피처로 LSTM 형태소 분석기 보강하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 위의 예제에서 각 단어는 시퀀스 모델에 대한 입력으로 사용되도록 임베딩 합니다.
# 이번에는 단어를 구성하는 문자로 부터 파생된 표현으로 단어 임베딩을 보강해 
# 보겠습니다. 접미사 같은 문자 수준의 정보는 품사에 큰 영향을 미치기 때문에
# 큰 도움이 될 것이라 예상합니다. 예를 들어 *-ly* 접미사가 있는 단어는 거의 
# 영어에서는 부사(abverbs)로 태그가 지정됩니다.
# 
# 우선 :math:`c_w`를 단어 :math:`w`의 문자 수준 표현이라 하겠습니다. 
# :math:`x_w`는 이전과 같이 단어 임베딩이라 하겠습니다. 그러면 시퀀스 모델에
# 대해서 :math:`x_w`와 :math:`c_w`이 연결되어 입력됩니다.
# 따라서 :math:`x_w`이 5차원이고 :math:`c_w`이 3차원이라면 LSTM은 8차원의
# 입력을 받습니다.
# 
# 문자 수준 표현을 얻기 위해서 단어 내 문자에 대해 LSTM을 이용하고, 
# :math:`c_w`을 이 LSTM의 최종 은닉 상태(hidden state)로 둡니다.
# 힌트:
#
# * 새 모델에는 두 개의 LSTM이 있습니다.
#   품사 태그 스코어를 출력하는 것 및 각 단어에 대해 문자 수준 표현을 출력하는 것
#   두 가지 입니다.
# * 문자에 대한 시퀀스 모델을 이용하려면 문자를 임베딩해야 합니다. 문자 임베딩은
#   character LSTM에 대한 입력이 됩니다.
#
