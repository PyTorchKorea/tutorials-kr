# -*- coding: utf-8 -*-
r"""
단어 임베딩: 어휘의 의미를 인코딩하자
===========================================

단어 임베딩이란 코퍼스 내 각 단어에 일대일로 대응하는 농밀한 실수 벡터의 집합, 혹은 이 벡터를
구하는 행위를 가리킵니다. 먼저, 자연어 처리에서는 주로 단어를 피처로 사용합니다. 이는 컴퓨터가
바로 이해하기는 상당히 어렵기 때문에 컴퓨터에게 친숙한 형태로 표현을 바꾸어 입력해야 합니다.
그렇다면, 단어를 어떻게 표현하는 것이 좋을까요? 물론 각 문자에 해당하는 ASCII코드를 사용할 수 있겠지만,
ASCII코드는 이 단어가 *무엇*인지를 알려줄 뿐, 단어가 어떤 *의미*를 가지는지는 알려주지 않습니다.
(룰베이스로 어미 등 문법적 특징을 활용하거나 영어의 경우 대문자를 사용할 수 있겠지만 충분하지 않습니다.)
단어를 어떻게 표현할지 뿐 아니라, 이 표현법을 어떠한 방식으로 연산해야 할 지 또한 큰 문제입니다.
보통 이러한 밀도 높은 벡터를 얻기 위해 사용하는 뉴럴넷 모델은 :math:`|V|`(코퍼스의 단어 개수)의
큰 인풋 차원과 몇 안되는 (텍스를 분류하는 문제라고 할 경우) 작은 아웃풋 차원을 가집니다.
즉, 단어들 간의 연산이 필수입니다. 어떻게 이 큰 차원의 공간을 작은 공간으로 변형시킬 수 있을까요?

자 먼저, 상기한 ASCII코드 대신 원핫 인코딩을 사용해보는 것은 어떨까요? 원핫 인코딩이란
하나의 단어 :math:`w`를 아래의 벡터로 표현하는 것을 말합니다.

.. math::  \overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}

여기서 1은 해당 벡터가 표현하고자 하는 단어에 해당하는 위치 1곳에 자리합니다. (나머지는 전부
0입니다.) 다른 단어를 나타내는 벡터에선 1이 다른 곳에 위치해 있겠죠.

원핫 인코딩은 만들기가 쉽다는 장점이 있지만, 단순한 만큼 단점도 있습니다. 일단 모든 단어를
표현할 수 있을 만한 크기가 되어야 합니다. 우리가 얼마나 많은 종류의 단어를 사용하는지를 생각
한다면 어마어마하게 큰 벡터라는 것을 알 수 있죠. 이 뿐만이 아닙니다. 원핫벡터는 모든 단어를
독립적인 개체로 가정하는 것을 볼 수 있습니다. 즉, 공간에서 봤을 때 완전히 다른 축에 위치해
있어서 단어간의 관계를 나타낼 수가 없습니다. 하지만 우리는 단어 사이의 *유사도*를 어떻게든
계산하고 싶은거죠. 왜 유사도가 중요하냐구요? 다음 예제를 봅시다.

우리의 목표가 언어 모델을 만드는 것이라고 가정하고 다음의 문장이 학습 데이터로써 주어졌다고 해봅시다.

* 수학자가 가게로 뛰어갔다.
* 물리학자가 가게로 뛰어갔다.
* 수학자가 리만 가설을 증명했다.

또한 학습 데이터에는 없는 아래 문장이 있다고 생각해봅시다.

* 물리학자가 리만가설을 증명했다.

ASCII 코드나 원핫 인코딩 기반 언어 모델은 위 문장을 어느정도 다룰 수 있겠지만, 개선의 여지가 있지 않을까요?
먼저 아래의 두 사실을 생각해봅시다.

* '수학자'와 '물리학자'가 문장 내에서 같은 역할을 맡고 있습니다. 이 두 단어는 어떻게든 의미적인 연관성이 있을 겁니다.
* 새로운 문장에서 '물리학자'가 맡고 있는 역할을 '수학자'가 맡고 있는 것을 학습 데이터에서 본 적이 있습니다.

우리 모델이 위의 사실을 사용해 '물리학자'가 새 문장에 잘 들어맞는다는 것을 추론할 수 있다면 참 좋을 것입니다.
이것이 위에서 언급한 유사도의 의미 입니다. 철자적 유사도 뿐 아니라 *의미적 유사도*인 것입니다.
이것이야말로 언어 데이터에 내재하는 희박성 (sparsity)에 대한 처방이 될 것입니다.
우리가 본 것과 아직 보지 않은 것 사이를 이어주는 것이죠.
앞으로는 다음의 언어학적 기본 명제를 가정하도록 합시다.
바로 비슷한 맥락에서 등장하는 단어들은 서로 의미적 연관성을 가진다는 것입니다.
언어학적으로는 '분산의미가설' 혹은 `distributional
hypothesis <https://en.wikipedia.org/wiki/Distributional_semantics>`__라고 합니다.


농밀한 단어 임베딩을 구해보자
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

어떻게 단어의 의미적 유사도를 인코딩 할 수 있을까요? 다시 말해, 어떻게 해야
벡터들 사이에 그에 해당하는 단어들이 가지는 유사도를 반영할 수 있을까요?
단어 데이터에 의미적 속성(attribute)을 부여하는 건 어떤가요? 예를 들어 '수학자'와 '물리학자'가
모두 뛸 수 있다면, 해당 단어의 '뛸 수 있음' 속성에 높은 점수를 주는 겁니다.
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

:math:`\phi`는 두 벡터 사이의 각입니다. 이런 식이면 정말 비슷한 단어는 유사도 1을 갖고,
정말 다른 단어는 유사도 -1을 갖겠죠. 비슷한 의미를 가질수록 같은 방향을 가리키고 있을 테니까요.


이 글 초반에 나온 희박한 원핫 벡터가 사실은 우리가 방금 정의한 의미 벡터의
특이 케이스라는 것을 금방 알 수 있습니다. 단어 벡터의 각 원소는 그 단어의 의미적 속성을
표현하고, 모든 단어 쌍의 유사도는 0이기 때문이죠. 위에서 정의한 의미 벡터는 *농밀* 합니다.
즉, 원핫 벡터에 비해 벡터를 이루는 원소의 값 중에 0이 적다고 할 수 있습니다.

하지만 이 벡터들은 구하기가 진짜 어렵습니다. 단어간의 유사도를 결정 지을 만한
의미적 속성은 어떻게 결정할 것이며, 속성을 결정했다고 하더라도 각 속성에
해당하는 값은 도대체 어떠한 기준으로 정해야 할까요? 속성과 속성값을 데이터에 기반해
만들고 자동으로 단어 벡터를 만들 수는 없을까요? 바로 그것이 딥러닝이 단어 임베딩에
필요한 부분입니다. 딥러닝은 인공신경망을 이용하여 사람의 개입 없이 속성의 표현 방법을
자동으로 학습합니다. 이를 이용해 단어 벡터를 모델 파라미터로 설정하고 모델 학습시에
단어 벡터도 함께 업데이트 되도록 하면 될 것입니다. 우리 신경망 모델이 적어도 이론상으로는
충분히 학습할 수 있는 *잠재 의미 속성*을 찾을 것입니다. 여기서 말하는 잠재 의미 속성으로
이루어진 벡터는 사람이 해석하기 상당히 어렵다는 점을 기억해 두세요. 위에서 수학자와 물리학자에게
커피를 좋아한다는 등 사람이 임의적으로 단어에 부여한 속성과는 달리, 인공신경망이 자동으로
단어의 속성을 찾는다면 그 속성과 값이 의미하는 바를 알기가 어려울 것입니다. 예를 들어서
신경망 모델이 찾은 '수학자'와 '물리학자'의 표현 벡터 둘 다 두번째 원소가 크다고 가정해 봅시다.
둘이 비슷하다는 건 알겠지만, 도대체 두번째 원소가 무엇을 의미하는지는 알기가 매우 힘든 것입니다.
표현 벡터 공간상에서 비슷하다는 정보 외에는 우리에게 아마 많은 정보를 주긴 어려울 것입니다.


요약하자면, **단어 임베딩은 단어의 *의미*를 표현하는 방법이며, 차후에 임베딩을 사용해서
풀고자 하는 문제에 유용할 의미 정보를 효율적으로 인코딩한 것입니다.** 품사 태그, 파스 트리 등
단어의 의미 외에 다른 것도 인코딩 할 수 있습니다! 피처 임베딩의 개념을 잡는 것이 중요합니다.


파이토치에서 단어 임베딩을 해보자
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we get to a worked example and an exercise, a few quick notes
about how to use embeddings in Pytorch and in deep learning programming
in general. Similar to how we defined a unique index for each word when
making one-hot vectors, we also need to define an index for each word
when using embeddings. These will be keys into a lookup table. That is,
embeddings are stored as a :math:`|V| \times D` matrix, where :math:`D`
is the dimensionality of the embeddings, such that the word assigned
index :math:`i` has its embedding stored in the :math:`i`'th row of the
matrix. In all of my code, the mapping from words to indices is a
dictionary named word\_to\_ix.

The module that allows you to use embeddings is torch.nn.Embedding,
which takes two arguments: the vocabulary size, and the dimensionality
of the embeddings.

To index into this table, you must use torch.LongTensor (since the
indices are integers, not floats).

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
# An Example: N-Gram Language Modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that in an n-gram language model, given a sequence of words
# :math:`w`, we want to compute
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# Where :math:`w_i` is the ith word of the sequence.
#
# In this example, we will compute the loss function on some training
# examples and update the parameters with backpropagation.
#

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
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
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

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
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!


######################################################################
# Exercise: Computing Word Embeddings: Continuous Bag-of-Words
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep
# learning. It is a model that tries to predict words given the context of
# a few words before and a few words after the target word. This is
# distinct from language modeling, since CBOW is not sequential and does
# not have to be probabilistic. Typcially, CBOW is used to quickly train
# word embeddings, and these embeddings are used to initialize the
# embeddings of some more complicated model. Usually, this is referred to
# as *pretraining embeddings*. It almost always helps performance a couple
# of percent.
#
# The CBOW model is as follows. Given a target word :math:`w_i` and an
# :math:`N` context window on each side, :math:`w_{i-1}, \dots, w_{i-N}`
# and :math:`w_{i+1}, \dots, w_{i+N}`, referring to all context words
# collectively as :math:`C`, CBOW tries to minimize
#
# .. math::  -\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)
#
# where :math:`q_w` is the embedding for word :math:`w`.
#
# Implement this model in Pytorch by filling in the class below. Some
# tips:
#
# * Think about which parameters you need to define.
# * Make sure you know what shape each operation expects. Use .view() if you need to
#   reshape.
#

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # example
