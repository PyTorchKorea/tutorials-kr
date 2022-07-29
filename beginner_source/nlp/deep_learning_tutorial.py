# -*- coding: utf-8 -*-
r"""
PyTorch를 이용한 딥러닝
**************************
**번역**: `황성수 <https://github.com/adonisues>`_

딥러닝 블록 구축 : 아핀 맵(affine maps), 비선형성, 객체
==========================================================================

딥러닝은 영리한 방법으로 비선형성을 가진 선형성을 구성하는 것으로
이루어집니다. 비선형성의 도입은 강력한 모델을 가능하게 합니다.
이 섹션에서 이 핵심 구성 요소를 다루고, 객체 함수를 만들고, 어떻게
모델이 학습되지는 살펴봅시다.


아핀 맵
~~~~~~~~~~~

딥러닝의 핵심 작업자 중 하나는 아핀 맵 입니다.
이 함수 :math:`f(x)` 는 다음과 같습니다.

.. math::  f(x) = Ax + b

여기서 :math:`A` 는 행렬, :math:`x, b` 는 벡터 입니다.
여기서 학습되는 변수는 :math:`A` 와 :math:`b` 입니다.
종종 :math:`b` 는 *편향(Bias)* 이라 불립니다.


PyTorch 와 대부분의 다른 딥러닝 프레임워크들은 고전적인 선형 대수학과
조금 다르게 동작합니다. 입력의 열 대신에 행으로 매핑합니다.
즉 주어진 :math:`A` 에서 출력의 :math:`i` 번째 행은
입력의 :math:`i` 번째 행에 매핑되고 편향(Bias)을 더합니다.
아래 예시를 살펴보십시오.

"""

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


######################################################################

lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = torch.randn(2, 5)
print(lin(data))  # yes


######################################################################
# 비선형성
# ~~~~~~~~~
#
# 먼저 왜 비선형성이 필요한지 설명하는 다음 사실을 주목하십시오.
# :math:`f(x) = Ax + b` 와 :math:`g(x) = Cx + d` 두 개의 아핀맵이 있다고 가정합니다.
# :math:`f(g(x))` 는 무엇일까요?
#
# .. math::  f(g(x)) = A(Cx + d) + b = ACx + (Ad + b)
#
# :math:`AC` 는 행렬이고 :math:`Ad + b` 는 벡터이므로 아핀맵 구성은
# 아핀맵이 주어집니다.
#
# 이것으로부터, 신경망이 아핀 구성의 긴 체인이 되길 원한다면,
# 단일 아핀 맵을 작성하는 것보다 이것이 모델에 추가하는 새로운 힘이
# 없다는 것을 알 수 있습니다.
#
# 아핀 계층 사이에 만약 비선형성을 적용한다면
# 이것은 위 경우와 달리 더욱더 강력한 모델을 구축할 수 있습니다.
#
# 핵심적인 비선형성 :math:`\tanh(x), \sigma(x), \text{ReLU}(x)` 들이 가장
# 일반적입니다. 아마 의문이 생길겁니다 : "왜 이런 함수들이지? 나는 다른 많은
# 비선형성을 생각할 수 있는데". 그 이유는 그들이 변화도(gradient)를 계산하기 쉽고,
# 변화도 연산은 학습에 필수적이기 때문입니다.
# 예를 들어서
#
# .. math::  \frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
#
# 빠른 참고: AI 클래스에 대한 소개에서 일부 신경망을 배웠지만 :math:`\sigma(x)` 가 기본이었을 것입니다.
# 일반적으로 사람들은 실제로 그것을 사용하지 않고 피합니다.
# 이것은 변화도가 인수의 절댓값이 커짐에 따라 매우 빨리 *사라지기* 때문입니다.
# 작은 변화도는 학습하기 어렵다는 것을 의미합니다.
# 대부분의 사람들은 tanh 또는 ReLU를 기본값으로 사용합니다.
#

# Pytorch에서 대부분의 비선형성은 torch.functional에 있습니다 ( F 로 가져옵니다)
# 일반적으로 비선형성은 아핀맵과 같은 파라미터를 가지고 있지 않습니다.
# 즉, 학습 중에 업데이트되는 가중치가 없습니다.
data = torch.randn(2, 2)
print(data)
print(F.relu(data))


######################################################################
# Softmax 및 확률
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 함수 :math:`\text{Softmax}(x)` 또한 단지 비선형성 이지만, 일반적으로 네트워크에서
# 마지막으로 수행되는 작업이라는 점에서 특별합니다.
# 이는 실수의 벡터를 취하여 확률 분포를 반환하기 때문입니다.
# 정의는 다음과 같습니다. :math:`x` 는 실수 벡터(음수, 양수 , 제약 없음)라고 하면,
# i번째 구성 요소는 :math:`\text{Softmax}(x)` 는
#
# .. math::  \frac{\exp(x_i)}{\sum_j \exp(x_j)}
#
# 출력은 확률 분포라는 것이 분명해야합니다:
# 각 요소는 음수가 아니며 모든 구성 요소의 합은 1입니다.
#
# 모두 음수가 아니게 하기 위해서 입력에 요소 단위의 지수 연산자를 적용한 다음
# 정규화 상수로 나누는 것도 생각할 수 있습니다.
#

# Softmax 도 torch.nn.functional 에 있습니다.
data = torch.randn(5)
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())  # 확률 분포이기 때문에 합이 1 입니다!
print(F.log_softmax(data, dim=0))  # log_softmax 도 있습니다.


######################################################################
# 목적 함수(Objective Functions)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 목적 함수는 네트워크가 최소화하도록 학습되는 함수입니다
# ( *손실 함수* 또는 *비용 함수* 라고 함).
# 먼저 학습 인스턴스를 선택하고 신경망을 통해 실행한 다음 출력의 손실을 계산합니다.
# 그런 다음 손실 함수의 미분을 취함으로써 모델의 파라미터가 업데이트됩니다.
# 직관적으로 모델이 자신의 대답에 완전히 확신하고 대답이 잘못되면 손실이 높아집니다.
# 답변에 자신이 있고 답변이 맞으면 손실이 적습니다.
#
# 학습 예제에서 손실 함수를 최소화하려는 아이디어는
# 네트워크가 잘 일반화되고 개발자 세트, 테스트 세트 또는 프로덕션에서
# 나타나지 않았던 예제(unseen examples)에 대해 작은 손실을 가지기를 바랍니다.
# 손실 함수의 예로 *음의 로그 우도 손실(negative log likelihood loss)* 있습니다.
# 이 것은 다중 클래스 분류에서 매우 자주 사용되는 목적 함수입니다.
# 감독 다중 클래스 분류의 경우에는 올바른 출력(정답을 맞춘 출력)의 음의 로그 확률을
# 최소화하도록 네트워크를 교육하는 것을 의미합니다.
# (또는 이와 동등하게 올바른 출력의 로그 확률을 최대화하십시오)
#


######################################################################
# 최적화와 학습
# =========================
#
# 그럼 인스턴스에 대해 손실 함수를 계산할 수 있다는 것은 무엇입니까? 그걸 어떻게 할까요?
# 우리는 이전에 Tensor가 그것을 계산하는데 사용된 것들에 해당하는 변화도를
# 계산하는 방법을 알고 있다는 것을 보았습니다.
# 손실은 Tensor이기 때문에 그것을 계산하는데 사용된 모든 파라미터와 관련하여
# 변화도를 계산할 수 있습니다! 그런 다음 표준 변화도 업데이트를 수행 할 수 있습니다.
# :math:`\theta` 가 우리의 파라미터라고 합시다.
# :math:`L(\theta)` 는 손실 함수, 그리고 :math:`\eta` 는 양의 러닝 레이트 입니다. 그러면
#
# .. math::  \theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L(\theta)
#
# 이 기본적인 그레디언트 업데이트 이상의 것을 하기 위해서 많은 알고리즘과
# 시도되고 있는 활발한 연구들이 있습니다.
# 많은 시도들은 학습 시간에 일어나는 것에 기반한 러닝 레이트를 변경해봅니다.
# 당신이 정말 관심이 없다면 특별히 이들 알고리즘이 무엇을 하는지 걱정할
# 필요가 없습니다. Torch는 torch.optim  패키지에서 많은 것을 제공하며
# 완전히 공개되어 있습니다. 가장 단순한 변화도 업데이트 사용은
# 더 복잡한 알고리즘을 사용하는 것과 동일합니다.
# 다른 업데이트 알고리즘과 업데이트 알고리즘을 위한 다른 파라미터(다른 초기 러닝 레이트)를
# 시도해 보는 것은 네트워크의 성능을 최적화하는데 중요합니다.
# 종종 기본 SGD를 Adam 또는 RMSprop 으로 교체하는 것이 눈에 띄게 성능
# 향상 시킵니다.
#


######################################################################
# Pytorch 에서 네트워크 구성요소 생성하기
# ==========================================
#
# NLP에 초점을 맞추기 전에,  PyTorch에서 아핀 맵과 비선형성만을 사용하여
# 네트워크를 구축하는 주석 처리된 예제를 수행 할 수 있습니다.
# 또한 손실 함수를 계산하는 방법, PyTorch에 내장된 음의 로그 우도를 사용하는 방법,
# 역전파를 통해 매개 변수를 업데이트하는 방법을 볼 것입니다.
#
# 모든 네트워크 구성 요소는 nn.Module에서 상속 받아 forward() 메서드를 재정의해야합니다.
# 이것은 상용구에 관한 것입니다. nn.Module에서의 상속은 구성 요소에 기능을 제공합니다.
# 예를 들어 그것은 학습 가능한 파라미터를 추적하도록 만들고,
# ``.to(device)`` 로 CPU와 GPU 를 교환할수 있습니다.
# ``torch.device("cpu")`` 는 CPU 장치를 ``torch.device("cuda:0")`` 는 GPU 장치를 사용합니다.
#
# 희소한 Bag-of-Words Representation 을 받아서 두개의 레이블 "영어"와 "스페인어"의 확률 분포
# 출력하는 네트워크의 주석이 달린 예시를 작성해 봅시다.
# 이 모델은 단순한 논리 회귀 입니다.
#


######################################################################
# 예제: 논리 회귀 Bag-of-Words 분류기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 우리 모델은 희소한 BoW 표현을 레이블에 대한 로그 확률로 매핑합니다.
# 사전의 각 단어에 하나의 색인을 할당합니다.
# 예를 들어서 전체 사전이 각각 0과 1의 색인을 가진 두개의 단어 "hello" 와 "world" 라고 합시다.
# "hello hello hello hello" 문장의 BoW 벡터는 다음과 같습니다.
#
# .. math::  \left[ 4, 0 \right]
#
#  "hello world world hello"는 다음과 같습니다.
#
# .. math::  \left[ 2, 2 \right]
#
# 일반화 하면 다음과 같습니다.
#
# .. math::  \left[ \text{Count}(\text{hello}), \text{Count}(\text{world}) \right]
#
# 이 BOW 벡터를 :math:`x` 라하면 네트워크의 출력은 다음과 같습니다:
#
# .. math::  \log \text{Softmax}(Ax + b)
#
# 즉, 아핀맵에 입력을 주고 그 다음 Log Softmax 를 합니다.
#

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix 는 사전의 각 단어를 고유한 정수로 매핑하고
# 그것은 BoW 벡터에서 자신의 색인이 됩니다.
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):  # nn.Module로 부터 상속 받기 !

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module의 초기화 함수 호출.  Dont get confused by syntax,
        # 문법에 혼란스러워 하지 마시고 단지 항상 nn.Module 에서 수행하십시오.
        super(BoWClassifier, self).__init__()

        # 필요한 파라미터를 정의 하십시오. 이 경우에는 아핀 매핑의 매개 변수 인 A와 b가 필요합니다.
        # Torch는 아핀 맵을 제공하는 nn.Linear()를 정의합니다
        # 입력 차원이 vocab_size이고 출력이 num_labels 인 이유를 이해했는지 확인하십시오!
        self.linear = nn.Linear(vocab_size, num_labels)

        # 주의! 비선형성 Log Softmax에는 파라미터가 없습니다!
        # 그래서 여기에 대해 걱정할 필요가 없습니다.

    def forward(self, bow_vec):
        # 선형 계층를 통해 입력을 전달한 다음 log_softmax로 전달합니다.
        # 많은 비선형성 및 기타 기능이 torch.nn.functional 에 있습니다

        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# 모델은 자신의 파라미터를 알고 있습니다. 아래에 있는 첫번째 출력은 A 두번째는 b 입니다.
# 모듈의 __init__ 함수에서 클래스 변수에 구성 요소를 할당 할 때마다 다음 행을 사용하여 완료합니다.
# self.linear = nn.Linear(...)
# 그런 다음 PyTorch 개발자의 Python 마법을 통해, 모듈(이 경우 BoWClassifier)은
# nn.Linear 파라미터에 대한 지식을 저장합니다
for param in model.parameters():
    print(param)

# 모델을 실행하려면 BoW 벡터를 전달합니다.
# 여기서 우리는 학습 할 필요가 없으므로 코드는 torch.no_grad()로 싸여 있습니다.
with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)
    print(log_probs)


######################################################################
# 위의 값 중 어느 것이 ENGLISH와 SPANISH의 로그 확률에 해당하는 값일까요?
# 우리는 정의하지 않았지만, 학습하기를 원한다면 필요합니다.
#

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}


######################################################################
# 그럼 학습을 해봅시다! 이를 위해 로그 확률을 얻고, 손실 함수를 계산하고,
# 손실 함수의 변화도를 계산한 다음 변화도 단계로 파라미터를
# 업데이트하기 위해 인스턴스를 통과시킵니다.  손실 기능은 nn 패키지의 Torch에서 제공합니다.
# nn.NLLLoss()는 원하는 음의 로그 우도 손실입니다. 또한 torch.optim에서 최적화 함수를 정의합니다.
# 여기서는 SGD 만 사용합니다.
#
# NLLLoss에 대한 *입력* 은 로그 확률의 벡터이고 목표 레이블입니다.
# 우리를 위한 로그 확률을 계산하지 않습니다. 이것이 네트워크의 마지막 계층이
# Log softmax 인 이유입니다. 손실 함수 nn.CrossEntropyLoss()는 Log softmax를 제외하고는 NLLLoss()와 같습니다.
#

# 훈련하기 전에 테스트 데이터를 실행하여 전후를 볼 수 있습니다.
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# "creo"에 해당하는 행렬 열을 인쇄하십시오.
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 일반적으로 교육 데이터를 여러 번 전달 합니다.
# 100은 실제 데이터 세트보다 훨씬 더 크지 만 실제 데이터 세트는 두 개 이상의 인스턴스를 가집니다.
# 일반적으로 5 ~ 30 개 에포크가 적당합니다.
for epoch in range(100):
    for instance, label in data:
        # 1 단계. PyTorch는 그라데이션을 축적합니다.
        # 각 인스턴스 전에 그들을 제거해야합니다.
        model.zero_grad()


        # 2 단계. BOW 벡터를 만들고 정수로 텐서로 대상을 싸야합니다.
        # 예를 들어, 대상이 SPANISH이면 정수 0으로 합니다.
        # 손실 함수는 로그 확률의 0번째 요소가 SPANISH에 해당하는 로그 확률임을 알 수 있습니다
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        # 3 단계. 순전파를 실행합니다.
        log_probs = model(bow_vec)

        # 4 단계. optimizer.step()을 호출하여 손실, 변화도를 계산하고 파라미터를 업데이트합니다.
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# 스페인어에 해당하는 색인이 올라갑니다. 영어가 내려갑니다!!
print(next(model.parameters())[:, word_to_ix["creo"]])


######################################################################
# 정답을 얻었습니다! 첫 번째 예제에서는 스페인어의 로그 확률이 ​​훨씬 높고
# 영어의 로그 확률은 테스트 데이터의 두 번째에서 훨씬 높다는 것을 알 수 있습니다.
#
# 이제 PyTorch 구성 요소를 만들고 이를 통해 일부 데이터를 전달하고
# 변화도 업데이트를 수행하는 방법을 살펴 보았습니다.
# 우리는 심도있는 NLP가 제공해야하는 것을 더 깊이 파고들 준비가되었습니다.
#
