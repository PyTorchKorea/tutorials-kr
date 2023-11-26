# -*- coding: utf-8 -*-
r"""
시퀀스 모델과 LSTM 네트워크
===================================================
**번역**: `박수민 <https://github.com/convin305>`_

지금까지 우리는 다양한 순전파(feed-forward) 신경망들을 보아 왔습니다. 
즉, 네트워크에 의해 유지되는 상태가 전혀 없다는 것입니다. 
이것은 아마 우리가 원하는 동작이 아닐 수도 있습니다. 
시퀀스 모델은 NLP의 핵심입니다. 이는 입력 간에 일종의 시간적 종속성이 존재하는 모델을 말합니다. 
시퀀스 모델의 고전적인 예는 품사 태깅을 위한 히든 마르코프 모델입니다. 
또 다른 예는 조건부 랜덤 필드입니다. 

순환 신경망은 일종의 상태를 유지하는 네트워크입니다. 
예를 들면, 출력은 다음 입력의 일부로 사용될 수 있습니다. 
정보는 네트워크가 시퀀스를 통과할 때 전파될 수 있습니다. 
LSTM의 경우에, 시퀀스의 각 요소에 대응하는 *은닉 상태(hidden state)* :math:`h_t` 가 존재하며,
이는 원칙적으로 시퀀스의 앞부분에 있는 임의 포인트의 정보를 포함할 수 있습니다. 
우리는 은닉 상태를 이용하여 언어 모델에서의 단어,
품사 태그 등 무수히 많은 것들을 예측할 수 있습니다. 


Pytorch에서의 LSTM
~~~~~~~~~~~~~~~~~

예제를 시작하기 전에, 몇 가지 사항을 유의하세요. 
Pytorch에서의 LSTM은 모든 입력이 3D Tensor 일 것으로 예상합니다. 
이러한 텐서 축의 의미는 중요합니다. 
첫 번째 축은 시퀀스 자체이고, 두 번째 축은 미니 배치의 인스턴스를 인덱싱하며, 
세 번째 축은 입력 요소를 인덱싱합니다. 
미니 배치에 대해서는 논의하지 않았으므로 이를 무시하고,
두 번째 축에 대해서는 항상 1차원만 가질 것이라고 가정하겠습니다. 
만약 우리가 "The cow jumped."라는 문장에 대해 시퀀스 모델을 실행하려면,
입력은 다음과 같아야 합니다. 

.. math::


   \begin{bmatrix}
   \overbrace{q_\text{The}}^\text{row vector} \\
   q_\text{cow} \\
   q_\text{jumped}
   \end{bmatrix}

다만, 사이즈가 1인 추가적인 2차원이 있다는 것을 기억해야 합니다. 

또한 한 번에 하나씩 시퀀스를 진행할 수 있으며,
이 경우 첫 번째 축도 사이즈가 1이 됩니다. 

간단한 예를 살펴보겠습니다. 
"""

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

######################################################################

lstm = nn.LSTM(3, 3)  # 입력 3차원, 출력 3차원
inputs = [torch.randn(1, 3) for _ in range(5)]  # 길이가 5인 시퀀스를 만듭니다

# 은닉 상태를 초기화합니다.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # 한 번에 한 요소씩 시퀀스를 통과합니다.
    # 각 단계가 끝나면, hidden에는 은닉 상태가 포함됩니다.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# 아니면 우리는 전체 시퀀스를 한 번에 수행할 수도 있습니다. 
# LSTM에 의해 반환된 첫 번째 값은 시퀀스 전체에 대한 은닉 상태입니다. 
# 두 번째는 가장 최근의 은닉 상태입니다. 
# (아래의 "hidden"과 "out"의 마지막 슬라이스(slice)를 비교해 보면 둘은 동일합니다.)
# 이렇게 하는 이유는 다음과 같습니다:
# "out"은 시퀀스의 모든 은닉 상태에 대한 액세스를 제공하고,
# "hidden"은 나중에 lstm에 인수 형태로 전달하여 
# 시퀀스를 계속하고, 역전파 하도록 합니다. 
# 추가로 두 번째 차원을 더합니다. 
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # 은닉 상태를 지웁니다.
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)


######################################################################
# 예시: 품사 태깅을 위한 LSTM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 섹션에서는 우리는 품사 태그를 얻기 위해 LSTM을 이용할 것입니다. 
# 비터비(Viterbi)나 순방향-역방향(Forward-Backward) 같은 것들은 사용하지 않을 것입니다. 
# 그러나 (도전적인) 연습으로, 어떻게 돌아가는지를 확인한 뒤에 
# 비터비를 어떻게 사용할 수 있는지에 대해서 생각해 보시기 바랍니다. 
# 이 예시에서는 임베딩도 참조합니다. 만약에 임베딩에 익숙하지 않다면, 
# `여기 <https://tutorials.pytorch.kr/beginner/nlp/word_embeddings_tutorial.html>`__.
# 에서 관련 내용을 읽을 수 있습니다.
#
# 모델은 다음과 같습니다. 단어가 :math:`w_i \in V` 일 때, 
# 입력 문장을 :math:`w_1, \dots, w_M` 라고 합시다. 또한, 
# :math:`T` 를 우리의 태그 집합라고 하고, :math:`w_i` 의 단어 태그를 :math:`y_i` 라고 합니다. 
# 단어 :math:`w_i` 에 대한 예측된 태그를 :math:`\hat{y}_i` 로 표시합니다. 
# 
#
# 이것은 :math:`\hat{y}_i \in T` 일 때, 출력이 :math:`\hat{y}_1, \dots, \hat{y}_M` 시퀀스인
# 구조 예측 모델입니다. 
#
# 예측을 하기 위해, LSTM에 문장을 전달합니다. 한 시간 단계
# :math:`i` 의 은닉 상태는 :math:`h_i` 로 표시합니다. 또한 각 태그에
# 고유한 인덱스를 할당합니다 (단어 임베딩 섹션에서 word\_to\_ix 를 사용한 것과 유사합니다.)
# 그러면 :math:`\hat{y}_i`  예측 규칙은 다음과 같습니다. 
#
# .. math::  \hat{y}_i = \text{argmax}_j \  (\log \text{Softmax}(Ah_i + b))_j
#
# 즉, 은닉 상태의 아핀 맵(affine map)에 대해 로그 소프트맥스(log softmax)를 취하고,
# 예측된 태그는 이 벡터에서 가장 큰 값을 가지는 태그가 됩니다. 
# 이것은 곧 :math:`A` 의 타깃 공간의 차원이 :math:`|T|` 라는 것을 
# 의미한다는 것을 알아두세요.
#
#
# 데이터 준비:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    # 태그는 다음과 같습니다: DET - 한정사;NN - 명사;V - 동사
    # 예를 들어, "The" 라는 단어는 한정사입니다.
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
# training_data의 각 튜플에 있는 각 단어 목록(문장) 및 태그 목록에 대해
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word는 아직 번호가 할당되지 않았습니다
            word_to_ix[word] = len(word_to_ix)  # 각 단어에 고유한 번호 할당
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # 각 태그에 고유한 번호 할당

# 이것들은 일반적으로 32나 64차원에 가깝습니다. 
# 훈련할 때 가중치가 어떻게 변하는지 확인할 수 있도록, 작게 유지하겠습니다. 
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

######################################################################
# 모델 생성:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM은 단어 임베딩을 입력으로 받고, 
        # 차원이 hidden_dim인 은닉 상태를 출력합니다. 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # 은닉 상태 공간에서 태그 공간으로 매핑하는 선형 레이어
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# 모델 학습:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 훈련 전의 점수를 확인하세요.
# 출력의 i,j요소는 단어 i에 대한 태그 j의 점수입니다.
# 여기서는 훈련을 할 필요가 없으므로, 코드는 torch.no_grad()로 래핑 되어 있습니다.
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  # 다시 말하지만, 일반적으로 300에폭을 수행하지는 않습니다. 이건 장난감 데이터이기 때문입니다.
    for sentence, tags in training_data:
        # 1단계, Pytorch는 변화도를 축적한다는 것을 기억하세요. 
        # 각 인스턴스 전에 이를 지워줘야 합니다. 
        model.zero_grad()

        # 2단계, 네트워크에 맞게 입력을 준비시킵니다. 
        # 즉, 입력들을 단어 인덱스들의 텐서로 변환합니다. 
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # 3단계, 순전파 단계(forward pass)를 실행합니다.
        tag_scores = model(sentence_in)

        # 4단계, 손실과 기울기를 계산하고, optimizer.step()을 호출하여 
        # 매개변수를 업데이트합니다. 
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# 훈련 후의 점수를 확인해 보세요. 
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # 문장은 "the dog ate the apple"입니다. i와 j는 단어 i에 대한 태그 j의 점수를 의미합니다.
    # 예측된 태그는 가장 점수가 높은 태그입니다.
    # 자, 아래의 예측된 순서가 0 1 2 0 1이라는 것을 확인할 수 있습니다.
    # 0은 1행에 대한 최댓값이므로, 
    # 1은 2행에 대한 최댓값이 되는 식입니다.
    # DET NOUN VERB DET NOUN은 올바른 순서입니다!
    print(tag_scores)


######################################################################
# 연습 : 문자-단위 특징과 LSTM 품사 태거 증강
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 위의 예제에서, 각 단어는 시퀀스 모델에 입력 역할을 하는 임베딩을 가집니다. 
# 단어의 문자에서 파생된 표현으로 단어 임베딩을 증가시켜보겠습니다. 
# 접사(affixes)와 같은 문자 수준의 정보는 품사에 큰 영향을 미치기 때문에, 
# 상당한 도움이 될 것으로 예상합니다. 
# 예를 들어, 접사 *-ly* 가 있는 단어는
# 영어에서 거의 항상 부사로 태그가 지정됩니다.
#
# 이것을 하기 위해서, :math:`c_w` 를 단어 :math:`w` 의 C를 단어 w의 문자 수준 표현이라고 하고, 
# 전과 같이 :math:`x_w` 를 단어임베딩이라고 합시다. 
# 그렇다면 우리의 시퀀스 모델에 대한 입력은 :math:`x_w` 와
# :math:`c_w` 의 연결이라고 할 수 있습니다. 만약에 :math:`x_w` 가 차원 5를 가지고, :math:`c_w`
# 차원 3을 가지면 LSTM은 차원 8의 입력을 받아들여야 합니다. 
#
# 문자 수준의 표현을 얻기 위해서, 단어의 문자에 대해서 LSTM을 수행하고
# :math:`c_w` 를 LSTM의 최종 은닉 상태가 되도록 합니다. 
# 힌트:
#
# * 새 모델에는 두 개의 LSTM이 있을 것입니다. 
#   POS 태그 점수를 출력하는 원래의 LSTM과 
#   각 단어의 문자 수준 표현을 출력하는 새로운 LSTM입니다. 
# * 문자에 대해 시퀀스 모델을 수행하려면, 문자를 임베딩해야 합니다. 
#   문자 임베딩은 문자 LSTM에 대한 입력이 됩니다.
#
