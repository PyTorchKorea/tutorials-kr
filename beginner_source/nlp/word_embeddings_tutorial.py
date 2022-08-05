# -*- coding: utf-8 -*-
r"""
단어 임베딩: 어휘의 의미(Lexical semantics)를 인코딩하기
======================================================
**번역**: `임성연 <http://github.com/sylim2357>`_

단어 임베딩(word embedding)이란 말뭉치(혹은 코퍼스, corpus) 내 각 단어에 일대일로 대응하는 밀집된 실수 벡터(dense vector)의 집합, 혹은 이 벡터를
구하는 행위를 가리킵니다. 주로 단어를 피처(feature)로 사용하는 자연어 처리 분야에서는 단어를 컴퓨터 친화적인
형태로 바꾸어 주는 작업이 필수적입니다. 컴퓨터가 단어를 바로 이해하기는 상당히 어렵기 때문이죠.
그렇다면, 단어를 어떻게 표현하는 것이 좋을까요? 물론 각 문자에 해당하는 ASCII코드를 사용할 수 있겠지만,
ASCII코드는 이 단어가 *무엇* 인지를 알려줄 뿐, 단어가 어떤 *의미* 를 가지는지는 알려주지 않습니다.
(룰베이스로 어미 등 문법적 특징을 활용하거나 영어의 경우 대문자를 사용할 수 있겠지만 충분하지 않습니다.)
단어를 어떻게 표현할지 뿐 아니라, 이 표현법을 어떠한 방식으로 연산해야 할지 또한 큰 문제입니다.
보통 이러한 밀도 높은 벡터를 얻기 위해 사용하는 뉴럴넷 모델은 :math:`|V|` (말뭉치의 단어 개수)의
큰 입력 차원과 몇 안되는 (텍스를 분류하는 문제라고 할 경우) 작은 출력 차원을 가집니다.
즉, 단어들 간의 연산이 필수입니다. 어떻게 이 큰 차원의 공간을 작은 공간으로 변형시킬 수 있을까요?

먼저, 상기한 ASCII코드 대신 원핫 인코딩(one-hot encoding)을 사용해보는 것은 어떨까요? 원핫 인코딩이란
하나의 단어 :math:`w` 를 아래의 벡터로 표현하는 것을 말합니다.

.. math::  \overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}

여기서 1은 해당 벡터가 표현하고자 하는 단어에 해당하는 위치 1곳에 자리합니다. (나머지는 전부
0입니다.) 다른 단어를 나타내는 벡터에선 1이 다른 곳에 위치해 있겠죠.

원핫 인코딩은 만들기가 쉽다는 장점이 있지만, 단순한 만큼 단점도 있습니다. 일단 단어 벡터 한 개는
모든 단어를 표현할 수 있을 만한 크기가 되어야 합니다. 우리가 얼마나 많은 종류의 단어를
사용하는지를 생각 한다면 어마어마하게 큰 벡터라는 것을 알 수 있죠. 이 뿐만이 아닙니다.
원핫 벡터는 모든 단어를 독립적인 개체로 가정하는 것을 볼 수 있습니다. 즉, 공간상에서
완전히 다른 축에 위치해 있어서 단어간의 관계를 나타낼 수가 없습니다. 하지만 우리는 단어
사이의 *유사도* 를 어떻게든 계산하고 싶은거죠. 왜 유사도가 중요하냐구요? 다음 예제를 봅시다.

우리의 목표가 언어 모델을 만드는 것이라고 가정하고 다음의 문장이 학습 데이터로써 주어졌다고 해봅시다.

* 수학자가 가게로 뛰어갔다.
* 물리학자가 가게로 뛰어갔다.
* 수학자가 리만 가설을 증명했다.

또한 학습 데이터에는 없는 아래 문장이 있다고 생각해봅시다.

* 물리학자가 리만 가설을 증명했다.

ASCII 코드나 원핫 인코딩 기반 언어 모델은 위 문장을 어느정도 다룰 수 있겠지만, 개선의 여지가 있지 않을까요?
먼저 아래의 두 사실을 생각해봅시다.

* '수학자'와 '물리학자'가 문장 내에서 같은 역할을 맡고 있습니다. 이 두 단어는 어떻게든 의미적인 연관성이 있을 겁니다.
* 새로운 문장에서 '물리학자'가 맡은 역할을 '수학자'가 맡는 것을 학습 데이터에서 본 적이 있습니다.

우리 모델이 위의 사실을 통해 '물리학자'가 새 문장에 잘 들어 맞는다는 것을 추론할 수
있다면 참 좋을 것입니다. 이것이 위에서 언급한 유사도의 의미입니다. 철자적 유사도 뿐
아니라 *의미적 유사도* 인 것입니다. 이것이야말로 언어 데이터에 내재하는 희박성(sparsity)에
대한 처방이 될 것입니다. 우리가 본 것과 아직 보지 않은 것 사이를 이어주는 것이죠.
앞으로는 다음의 언어학적 기본 명제를 가정하도록 합시다. 바로 비슷한 맥락에서 등장하는
단어들은 서로 의미적 연관성을 가진다는 것입니다. 언어학적으로는 `분산 의미 가설(distributional
hypothesis) <https://en.wikipedia.org/wiki/Distributional_semantics>`__ 이라고도 합니다.


밀집된 단어 임베딩 구하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

어떻게 단어의 의미적 유사도를 인코딩 할 수 있을까요? 다시 말해, 어떻게 해야 단어의 유사도를
단어 벡터에 반영할 수 있을까요? 단어 데이터에 의미적 속성(attribute)을 부여하는 건 어떤가요?
예를 들어 '수학자'와 '물리학자'가 모두 뛸 수 있다면, 해당 단어의 '뛸 수 있음' 속성에 높은 점수를 주는 겁니다.
계속 해봅시다. 다른 단어들에 대해서는 어떠한 속성을 만들 수 있을지 생각해봅시다.

만약 각 속성을 하나의 차원이라고 본다면 하나의 단어에 아래와 같은 벡터를 배정할 수 있을겁니다.

.. math::

    q_\text{수학자} = \left[ \overbrace{2.3}^\text{뛸 수 있음},
   \overbrace{9.4}^\text{커피를 좋아함}, \overbrace{-5.5}^\text{물리 전공임}, \dots \right]

.. math::

    q_\text{물리학자} = \left[ \overbrace{2.5}^\text{뛸 수 있음},
   \overbrace{9.1}^\text{커피를 좋아함}, \overbrace{6.4}^\text{물리 전공임}, \dots \right]

그러면 아래와 같이 두 단어 사이의 유사도를 구할 수 있습니다. ('유사도'라는 함수를 정의하는 겁니다)

.. math::  \text{유사도}(\text{물리학자}, \text{수학자}) = q_\text{물리학자} \cdot q_\text{수학자}

물론 보통은 이렇게 벡터의 길이로 나눠주지만요.

.. math::

    \text{유사도}(\text{물리학자}, \text{수학자}) = \frac{q_\text{물리학자} \cdot q_\text{수학자}}
   {\| q_\text{물리학자} \| \| q_\text{수학자} \|} = \cos (\phi)

:math:`\phi` 는 두 벡터 사이의 각입니다. 이런 식이면 정말 비슷한 단어는 유사도 1을 갖고,
정말 다른 단어는 유사도 -1을 갖겠죠. 비슷한 의미를 가질수록 같은 방향을 가리키고 있을 테니까요.

이 글 초반에 나온 희박한 원핫 벡터가 사실은 우리가 방금 정의한 의미 벡터의
특이 케이스라는 것을 금방 알 수 있습니다. 단어 벡터의 각 원소는 그 단어의 의미적 속성을
표현하고, 모든 단어 쌍의 유사도는 0이기 때문이죠. 위에서 정의한 의미 벡터는 *밀집* 되어 있습니다.
즉, 원핫 벡터에 비해 0 원소의 수가 적다고 할 수 있습니다.

하지만 이 벡터들은 구하기가 진짜 어렵습니다. 단어간의 유사도를 결정 지을 만한
의미적 속성은 어떻게 결정할 것이며, 속성을 결정했다고 하더라도 각 속성에
해당하는 값은 도대체 어떠한 기준으로 정해야 할까요? 속성과 값을 데이터에 기반해
만들고 자동으로 단어 벡터를 만들 수는 없을까요? 있습니다. 딥러닝을 사용하면 말이죠.
딥러닝은 인공신경망을 이용하여 사람의 개입 없이 속성의 표현 방법을 자동으로 학습합니다.
이를 이용해 단어 벡터를 모델 모수로 설정하고 모델 학습시에 단어 벡터도 함께 업데이트 하면
될 것입니다. 이렇게 우리 신경망 모델은 적어도 이론상으로는 충분히 학습할 수 있는
*잠재 의미 속성* 을 찾을 것입니다. 여기서 말하는 잠재 의미 속성으로 이루어진 벡터는 사람이
해석하기 상당히 어렵다는 점을 기억해 두세요. 위에서 수학자와 물리학자에게 커피를 좋아한다는
등 사람이 임의적으로 단어에 부여한 속성과는 달리, 인공신경망이 자동으로 단어의 속성을 찾는다면
그 속성과 값이 의미하는 바를 알기가 어려울 것입니다. 예를 들어서 신경망 모델이 찾은 '수학자'와
'물리학자'의 표현 벡터 둘 다 두번째 원소가 크다고 가정해 봅시다. 둘이 비슷하다는 건 알겠지만,
도대체 두번째 원소가 무엇을 의미하는지는 알기가 매우 힘든 것입니다. 표현 벡터 공간상에서
비슷하다는 정보 외에는 아마 많은 정보를 주긴 어려울 것입니다.

요약하자면, **단어 임베딩은 단어의 *의미* 를 표현하는 방법이며, 차후에 임베딩을 사용해서
풀고자 하는 문제에 유용할 의미 정보를 효율적으로 인코딩한 것입니다.** 품사 태그, 파스 트리(parse tree) 등
단어의 의미 외에 다른 것도 인코딩 할 수 있습니다! 피처 임베딩의 개념을 잡는 것이 중요합니다.


파이토치에서 단어 임베딩 하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

실제로 코드와 예시를 보기 전에, 파이토치를 비롯해 딥러닝 관련 프로그래밍을 할 때
단어 임베딩을 어떻게 사용하는지에 대해 조금 알아봅시다. 맨 위에서 원핫 벡터를
정의했던 것 처럼, 단어 임베딩을 사용할 때에도 각 단어에 인덱스를 부여해야 합니다.
이 인덱스를 참조 테이블(look-up table)에서 사용할 것입니다. 즉, :math:`|V| \times D` 크기의 행렬에
단어 임베딩을 저장하는데, :math:`D` 차원의 임베딩 벡터가 행렬의 :math:`i` 번째 행에
저장되어있어 :math:`i` 를 인덱스로 활용해 임베딩 벡터를 참조하는 것입니다.
아래의 모든 코드에서는 단어와 인덱스를 매핑해주는 딕셔너리를 word\_to\_ix라 칭합니다.

파이토치는 임베딩을 손쉽게 사용할 수 있게 torch.nn.Embedding에 위에서 설명한 참조 테이블
기능을 지원합니다. 이 모듈은 단어의 개수와 임베딩의 차원, 총 2개의 변수를 입력 변수로 받습니다.

torch.nn.Embedding 테이블의 임베딩을 참조하기 위해선 torch.LongTensor 타입의 인덱스 변수를
꼭 사용해야 합니다. (인덱스는 실수가 아닌 정수이기 때문입니다.)

"""

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

######################################################################

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)


######################################################################
# 예시: N그램 언어 모델링
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# N그램 언어 모델링에선 단어 시퀀스 :math:`w` 가 주어졌을 때 아래의 것을 얻고자
# 함을 상기해 봅시다.
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# :math:`w_i` 는 시퀀스에서 i번째 단어입니다.
#
# 이 예시에서는 학습 데이터를 바탕으로 손실 함수를 계산하고 역전파를 통해
# 모수를 업데이트 해보겠습니다.
#

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# 셰익스피어 소네트(Sonnet) 2를 사용하겠습니다.
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# 원래는 입력을 제대로 토큰화(tokenize) 해야하지만 이번엔 간소화하여 진행하겠습니다.
# 튜플로 이루어진 리스트를 만들겠습니다. 각 튜플은 ([ i-CONTEXT_SIZE 번째 단어, ..., i-1 번째 단어 ], 목표 단어)입니다.
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# 첫 3개의 튜플을 출력하여 데이터가 어떻게 생겼는지 보겠습니다.
print(ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in ngrams:

        # 첫번째. 모델에 넣어줄 입력값을 준비합니다. (i.e, 단어를 정수 인덱스로
        # 바꾸고 파이토치 텐서로 감싸줍시다.)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # 두번째. 토치는 기울기가 *누적* 됩니다. 새 인스턴스를 넣어주기 전에
        # 기울기를 초기화합니다.
        model.zero_grad()

        # 세번째. 순전파를 통해 다음에 올 단어에 대한 로그 확률을 구합니다.
        log_probs = model(context_idxs)

        # 네번째. 손실함수를 계산합니다. (파이토치에서는 목표 단어를 텐서로 감싸줘야 합니다.)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # 다섯번째. 역전파를 통해 기울기를 업데이트 해줍니다.
        loss.backward()
        optimizer.step()

        # tensor.item()을 호출하여 단일원소 텐서에서 숫자를 반환받습니다.
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # 반복할 떄마다 손실이 줄어드는 것을 봅시다!

# "beauty"와 같이 특정 단어에 대한 임베딩을 확인하려면,
print(model.embeddings.weight[word_to_ix["beauty"]])

######################################################################
# 예시: 단어 임베딩 계산하기: Continuous Bag-of-Words
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Continuous Bag-of-Words (CBOW) 모델은 NLP 딥러닝에서 많이 쓰입니다.
# 이 모델은 문장 내에서 주변 단어, 즉 앞 몇 단어와 뒤 몇 단어를 보고 특정
# 단어를 예측하는데, 언어 모델링과는 달리 순차적이지도 않고 확률적이지도 않습니다.
# 주로 CBOW는 복잡한 모델의 초기 입력값으로 쓰일 단어 임베딩을 빠르게 학습하는
# 데에 쓰입니다. 이것을 *사전 훈련된(pre-trained) 임베딩* 이라고 부르죠.
# 몇 퍼센트 정도의 성능 향상을 기대할 수 있는 기법입니다.
#
# CBOW 모델은 다음과 같습니다. 목표 단어 :math:`w_i` 와 그 양쪽에 :math:`N` 개의
# 문맥 단어 :math:`w_{i-1}, \dots, w_{i-N}` 와 :math:`w_{i+1}, \dots, w_{i+N}`
# 가 주어졌을 때, (문맥 단어를 총칭해 :math:`C` 라고 합시다.)
#
# .. math::  -\log p(w_i | C) = -\log \text{Softmax}\left(A(\sum_{w \in C} q_w) + b\right)
#
# 위 식을 최소화하는 것이 CBOW의 목적입니다. 여기서 :math:`q_w` 는 단어 :math:`w` 의
# 임베딩 입니다.
#
# 아래의 클래스 템플릿을 보고 파이토치로 CBOW를 구현해 보세요. 힌트는 다음과 같습니다.
#
# * 어떤 모수를 정의해야 하는지 생각해보세요.
# * 각 작업에서 다루어지는 변수의 차원이 어떤지 꼭 생각해보세요.
#   텐서의 모양을 바꿔야 한다면 .view()를 사용하세요.
#

CONTEXT_SIZE = 2  # 왼쪽으로 2단어, 오른쪽으로 2단어
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# 중복된 단어를 제거하기 위해 `raw_text` 를 집합(set) 자료형으로 바꿔줍니다.
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = (
        [raw_text[i - j - 1] for j in range(CONTEXT_SIZE)]
        + [raw_text[i + j + 1] for j in range(CONTEXT_SIZE)]
    )
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass

# 모델을 만들고 학습해 보세요.
# 아래는 데이터 준비를 원활하게 돕기 위한 함수입니다.


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # 예시
