# -*- coding: utf-8 -*-
r"""
심화 과정 : Bi-LSTM CRF와 동적 결정
======================================================

동적, 정적 딥 러닝 툴킷(toolkits) 비교
--------------------------------------------

Pytorch는 *동적* 신경망 툴킷입니다. 다른 동적 신경망 툴킷으로는
`Dynet <https://github.com/clab/dynet>`_ 이 있습니다.(이 툴킷을
예로 든 이유는 사용하는 법이 Pytorch와 비슷하기 때문입니다. Dynet의 예제를 보면
Pytorch로 구현할 때도 도움이 될 것입니다.) 반대로 *정적* 툴킷들로
Theano, Keras, TensorFlow 등이 있습니다. 주요 차이점은 다음과 같습니다:

* 정적 툴킷을 쓸 때는 계산 그래프를 한 번만 정의하고, 컴파일 한 후,
  데이터를 계산 그래프에 넘깁니다.
* 동적 툴킷에서는 *각 데이터* 의 계산 그래프를 정의하며 컴파일하지
  않고 즉각 실행됩니다.

경험이 많지 않다면 두 방식의 차이를 알기 어렵습니다. 딥 러닝 기반의
구구조 분석기(constituent parser)를 예로 들어보겠습니다. 모델은 대략
다음과 같은 과정을 수행합니다:

* 트리를 상향식(bottom-up)으로 만들어 나갑니다.
* 최상위 노드를 태깅합니다. (문장의 각 단어)
* 거기서부터 신경망과 단어들의 임베딩을 이용해 구구조를 이루는 조합을
  찾아냅니다. 새로운 구구조를 생성할 때마다 구구조의 임베딩을 얻기 위한
  어떤 기술이 필요합니다. 지금은 신경망이 오직 입력 문장만 참고할
  것입니다. "The green cat scratched the wall"이란 문장에서, 모델의 어느 시점에
  :math:`(i,j,r) = (1, 3, \text{NP})` 범위 (단어 1에서부터 단어 3까지가
  NP 구구조라는 뜻이며, 이 문장에서는 "The green cat") 를 결합하길 원할
  것입니다.

그런데, 또다른 문장 "Somewhere, the big fat cat scratched the wall" 에서는
어느 시점에 :math:`(2, 4, NP)` 구구조를 만들기를 원할 것입니다. 우리가
만들기 원하는 구구조들은 문장에 따라 다릅니다. 만약 정적 툴킷에서처럼
계산 그래프를 한 번만 컴파일한다면, 이 과정을 프로그래밍하기 매우 어렵거나
불가능할 것입니다. 하지만 동적 툴킷에서는 하나의 계산 그래프만 있지
않습니다. 각 문장들마다 새로운 계산 그래프가 있을 수 있기 때문에 이런
문제가 없습니다.

동적 틀킷은 디버깅 하기 더 쉽고, 코드가 기반 언어와 더 비슷합니다
(Pytorch와 Dynet이 Keras 또는 Theano 보다 Python 코드와 더 비슷합니다).

Bi-LSTM Conditional Rnadom Field 설명
-------------------------------------------

이 영역에서는 개체명 인식을 수행하는 완성된 Bi-LSTM Conditional Random
Field 예시를 살펴보겠습니다. 위에 나온 LSTM 태거(tagger)는 일반적으로
품사 태깅을 하기에 충분합니다. 하지만 CRF 같은 연속된 데이터를 다루는
모델은 좋은 개체명 인식 모델(NER)에 꼭 필요합니다. 여러분이 CRF를 잘 알고
있다고 가정하겠습니다. 이름이 무섭게 들릴 수도 있지만, LSTM이 특징을
제공하는 점을 제외하면 이 모델은 CRF 입니다. 하지만 더 발전된 모델이며,
이 튜토리얼의 앞부분에 나왔던 모델보다 훨씬 복잡합니다. 넘어가고 싶다면
넘어가도 괜찮습니다. 이해할 수 있다고 생각한다면, 아래를 읽어보세요:

-  태그 k에 대한 i번째 단계의 비터비(viterbi) 변수를 위해 순환 흐름을 만든다.
-  순방향 변수를 계산하기 위해 위의 순한 흐름을 조정한다.
-  순방향 변수를 로그 공간에서 계산하기 위해 다시 한 번 조정한다.
   (힌트 : 로그-합-지수승)

위의 세가지를 할 수 있다면, 아래의 코드를 이해할 수 있을 것입니다.
CRF는 조건부 확률을 계산한다는 점을 기억하세요. :math:`y` 를 연속된
태그라 하고, :math:`x` 를 연속된 입력 단어라 하겠습니다. 그러면 아래의
식을 계산할 수 있습니다.

.. math::  P(y|x) = \frac{\exp{(\text{Score}(x, y)})}{\sum_{y'} \exp{(\text{Score}(x, y')})}

점수(score) 함수는 아래와 같이 정의된 로그 포텐셜(potential) :math:`\log \psi_i(x,y)`
함수에 의해 결정됩니다.

.. math::  \text{Score}(x,y) = \sum_i \log \psi_i(x,y)

분배 함수(partition function)를 단순화하기 위해서, 포텐셜이 주변의
특징들만 반영한다고 하겠습니다.

Bi-LSTM CRF 안에 배출(emission), 전이(transition) 두 종류의 포텐셜을
정의합니다. :math:`i` 번째 단어에 대한 배출 포텐셜은 Bi-LSTM의
:math:`i` 번째 시점의 은닉 상태가 결정합니다. 전이 점수는 :math:`|T|x|T|`
형태인 행렬 :math:`\textbf{P}` 에 저장되어 있습니다. :math:`T` 는
태그의 집합입니다. 이 구현에서, :math:`\textbf{P}_{j,k}` 는 tag :math:`j` 에서
tag :math:`k` 로의 전이 점수를 의미합니다. 따라서:

.. math::  \text{Score}(x,y) = \sum_i \log \psi_\text{EMIT}(y_i \rightarrow x_i) + \log \psi_\text{TRANS}(y_{i-1} \rightarrow y_i)

.. math::  = \sum_i h_i[y_i] + \textbf{P}_{y_i, y_{i-1}}

두 번째 식에서 고유하고 음수가 아닌 인덱스에 의해 태그가 부여됐다고
간주합니다.

위의 설명이 너무 간단하다고 생각한다면 CRF에 대한 Michael Collins의
글을 `여기 <http://www.cs.columbia.edu/%7Emcollins/crf.pdf>`__ 에서
읽어보세요.

구현 문서
--------------------

아래의 예시는 로그 공간에서 분배 함수를 계산하기 위한 순방향 알고리즘과
복호화하기 위한 비터비 알고리즘을 구현한 것입니다. 역전파
단계에서 변화도는 자동으로 계산될 것입니다. 우리가 직접 할 일은
없습니다.

이 구현은 최적의 상태가 아닙니다. 과정을 이해했다면, 순방향 알고리즘
상에서 다음 태그를 순차적으로 처리하는 과정을 하나의 큰 연산으로 줄일
수 있다는 것을 아마 빠르게 알 수 있을 것입니다. 이 코드는 가능한 읽기
쉽게 작성했습니다. 적절하게 수정하면, 이 태거를 실제 문제들에 사용할
수도 있을 것입니다.
"""
# 작성자: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

#####################################################################
# 코드 가독성을 높여주는 보조 함수들


def argmax(vec):
    # argmax를 파이썬 정수형으로 반환합니다.
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# 순방향 알고리즘을 위해 수치적으로 안정적인 방법으로 로그 합 지수승을 계산합니다.
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#####################################################################
# 모델 생성


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # LSTM의 출력을 태그 공간으로 대응시킵니다.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 전이 매개변수 행렬. i, j 성분은 i에서 j로 변할 때의 점수입니다.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 이 두 코드는 시작 태그로 전이하지 않고, 정지 태그에서부터
        # 전이하지 않도록 강제합니다.
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # 분배 함수를 계산하기 위해 순방향 알고리즘을 수행합니다.
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG는 모든 점수를 갖고 있습니다.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 자동으로 역전파 되도록 변수로 감쌉니다.
        forward_var = init_alphas

        # 문장의 각 성분을 반복 처리합니다.
        for feat in feats:
            alphas_t = []  # 현재 시점의 순방향 텐서
            for next_tag in range(self.tagset_size):
                # 이전의 태그와 상관없이 배출 점수를 전파합니다.
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # trans_score의 i번째 성분은 i로부터 next_tag로 전이할 점수입니다.
                trans_score = self.transitions[next_tag].view(1, -1)
                # next_tag_var의 i번째 성분은 로그-합-지수승을 계산하기 전
                # i에서 next_tag로 가는 간선의 값입니다.
                next_tag_var = forward_var + trans_score + emit_score
                # 이 태그의 순방향 변수는 모든 점수들의 로그-합-지수승 계산
                # 결과입니다.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 주어진 태그 순열에 점수를 매깁니다.
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # 비터비 변수를 로그 공간 상에 초기화합니다.
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # i 단계의 forward_var는 i-1 단계의 비터비 변수를 갖고 있습니다.
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 현재 단계의 backpointer를 갖고 있습니다.
            viterbivars_t = []  # 현재 단계의 비터비 변수를 갖고 있습니다.

            for next_tag in range(self.tagset_size):
                # next_tag_var[i]는 이전 단계의 태그 i에 대한 비터비 변수와,
                # 태그 i에서 next_tag로 전이할 점수를 더한 값을 갖고 있습니다.
                # 배출 점수는 argmax와 상관 없기 때문에(아래 코드에서 추가할 것입니다)
                # 여기에 포함되지 않습니다.
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 이제 배출 점수를 더합니다. 그리고 방금 계산한 비터비 변수의
            # 집합을 forward_var에 할당합니다.
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # STAP_TAG로의 전이
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 최적의 경로를 구하기 위해 back pointer를 따라갑니다.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 시작 태그를 빼냅니다 (시작 태그는 반환된 필요가 없습니다)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 완결성 검사 (Sanity check)
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # 이 함수와 위의 _forward_alg를 헷갈리지 마세요.
        # Bi-LSTM으로부터 배출 점수를 얻습니다.
        lstm_feats = self._get_lstm_features(sentence)

        # 주어진 특징(배출 점수)들로 최적의 경로를 찾아냅니다.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#####################################################################
# 훈련 실행


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# 훈련용 데이터를 만듭니다.
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 훈련 전 예측 결과를 확인합니다.
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
# 위의 보조 함수 영역에 있는 prepare_sequence 함수가 불러와 졌는지 확인합니다.
for epoch in range(
        300):  # 다시 말하지만, 아마 300 에폭을 실행하진 않을 것입니다. 이것은 연습용 데이터입니다.
    for sentence, tags in training_data:
        # 1단계. Pytorch가 변화도를 누적한다는 것을 기억하세요.
        # 그것들을 제거합니다.
        model.zero_grad()

        # 2단계. 입력 데이터를 신경망에 사용될 수 있도록 단어
        # 인덱스들의 텐서로 변환합니다.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # 3단계. 순방향 계산을 수행합니다.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # 4단계. 손실값, 변화도를 계산하고 optimizer.step()을 호출하여
        # 매개변수들을 갱신합니다.
        loss.backward()
        optimizer.step()

# 훈련이 끝난 후 예측 결과를 확인합니다.
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# 다 했습니다!


######################################################################
# 연습 : 판별적(discriminative) 태깅을 위한 새로운 손실 함수
# ------------------------------------------------------------
#
# 사실 복호화 할 때는 비터비 경로 점수로 역전파를 하지 않기 때문에 계산
# 그래프를 만들 필요가 없었습니다. 그러나 이미 만들었으니, 비터비 경로
# 점수와 실제 정답 경로 점수의 차이를 손실 함수로 사용해서 태거를
# 학습시켜 보세요. 손실 함수의 값은 음수가 아니어야 하며, 예측된 태그
# 순열이 정답이라면 손실 함수의 값은 0이어야 합니다. 이것은 본질적으로
# *구조화된 퍼셉트론* 입니다.
#
# 이미 비터비와 score_sentence 함수가 구현되어 있기 때문에 간단히 수정할
# 수 있습니다. 이 모델은 *학습 데이터에 따라 변하는* 계산 그래프의 한
# 예시입니다. 이 모델을 정적 툴킷에서 구현해 보지는 않았는데, 구현이
# 가능하지만 덜 직관적일 수 있습니다.
#
# 실제 데이터를 사용해보고 비교해보세요!
