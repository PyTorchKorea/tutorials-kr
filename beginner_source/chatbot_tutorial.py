# -*- coding: utf-8 -*-

"""
챗봇 튜토리얼
================
**Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
  **번역**: `김진현 <https://github.com/lewha0>`_
"""


######################################################################
# 이 튜토리얼에서는 순환(recurrent) 시퀀스 투 시퀀스(sequence-to-sequence)
# 모델의 재미있고 흥미로운 사용 예를 살펴보려 합니다. 간단한 챗봇을 학습해
# 볼 텐데, 사용할 데이터는 영화 대본으로 구성된 `Cornell Movie-Dialogs(코넬
# 대학교의 영화 속 대화 말뭉치 데이터
# <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# 입니다.
#
# 대화형 모델은 많은 사람들이 관심을 갖는 인공지능 분야의 연구 주제입니다.
# 고객 서비스와 관련된 활용, 온라인 헬프데스크 등 여러 상황에서 챗봇을
# 활용할 수 있습니다. 많은 챗봇이 검색 기반(retrieval-based) 모델을
# 사용하는데, 이는 특정한 형식을 갖춘 질문에 대해 미리 정해진 반응을
# 출력하는 방식입니다. 분야를 특정 회사의 IT 헬프데스크처럼 한정짓는다면
# 이러한 모델을 사용해도 충분합니다. 그러나 이런 모델은 좀 더 일반적인
# 상황에 활용할 수 있을만큼 강력하진 않습니다. 기계를 학습시켜서 사람과
# 여러 주제에 대해 의미 있는 대화를 하게끔 하는 것은 아직 해결되지 않은
# 연구 주제입니다. 그러나 최근에 딥러닝이 유행하면서 여러 가지의 강력한
# 생성 모델이 등장했습니다. 그러한 모델의 한 예인 구글의 `신경 대화
# 모델(Neural Conversational Model) <https://arxiv.org/abs/1506.05869>`__ 은
# 다중 도메인 대화 생성 모델(multi-domain generative conversational models)
# 분야에 있어서 큰 진전을 이루었습니다. 우리는 이 튜토리얼을 통해 이러한
# 모델을 PyTorch로 구현해보려 합니다.
#
# .. figure:: /_static/img/chatbot/bot.png
#    :align: center
#    :alt: bot
#
# .. code:: python
#
#   > hello? (안녕하세요?)
#   Bot: hello . (안녕하세요.)
#   > where am I? (여긴 어디죠?)
#   Bot: you re in a hospital . (병원입니다.)
#   > who are you? (당신은 누구시죠?)
#   Bot: i m a lawyer . (변호사입니다.)
#   > how are you doing? (어떻게 지내세요?)
#   Bot: i m fine . (잘 지냅니다.)
#   > are you my friend? (당신은 제 친구인가요?)
#   Bot: no . (아뇨.)
#   > you're under arrest (당신을 체포하겠습니다)
#   Bot: i m trying to help you ! (난 당신을 도우려 하는 겁니다!)
#   > i'm just kidding (농담이었어요)
#   Bot: i m sorry . (미안하네요.)
#   > where are you from? (어디서 오셨어요?)
#   Bot: san francisco . (샌프란시스코요.)
#   > it's time for me to leave (전 이제 가봐야겠네요)
#   Bot: i know . (알겠습니다.)
#   > goodbye (안녕히 계세요)
#   Bot: goodbye . (안녕히 가세요.)
#
# **이 튜토리얼의 핵심 내용**
#
# -  `코넬 대학교의 영화 속 대화 말뭉치 데이터셋
#    <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__ 을
#    읽어오고 전처리합니다
# -  `Luong의 어텐션(attention) 메커니즘 <https://arxiv.org/abs/1508.04025>`__ 을
#    이용하여 sequence-to-sequence 모델을 구현합니다
# -  미니배치를 이용하여 인코더와 디코더를 함께 학습합니다
# -  탐욕적 탐색 기법(greedy-search)을 사용하는 디코더 모듈을 구현합니다
# -  학습한 챗봇과 대화를 나눠 봅니다
#
# **감사의 글**
#
# 이 튜토리얼은 다음 자료의 도움을 받아 작성하였습니다.
#
# 1) Yuan-Kuei Wu의 pytorch-chatbot 구현체:
#    https://github.com/ywk991112/pytorch-chatbot
#
# 2) Sean Robertson의 practical-pytorch seq2seq-translation 예제:
#    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
#
# 3) FloydHub의 코넬 대학교의 영화 말뭉치 데이터 전처리 코드:
#    https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
#


######################################################################
# 준비 단계
# ---------
#
# 시작에 앞서, `여기 <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__ 에서
# ZIP 파일 형태의 데이터를 내려받고, 현재 디렉토리 아래에 ``data/`` 라는
# 디렉토리를 만들어서 내려받은 데이터를 옮겨두시기 바랍니다.
#
# 그 다음에는, 몇 가지 필요한 도구들을 import 하겠습니다.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


######################################################################
# 데이터 읽기 & 전처리하기
# ------------------------
#
# 다음 단계는 데이터 파일의 형식을 재조정한 후, 우리가 작업하기 편한
# 구조로 읽어들이는 것입니다.
#
# `코넬 대학교의 영화 속 대화 말뭉치 데이터셋
# <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__ 은
# 영화 속 등장 인물의 대화가 풍부하게 포함된 데이터셋입니다.
#
# -  영화 속 등장 인물 10,292 쌍이 대화를 220,579번 주고받습니다
# -  영화 617개의 등장 인물 9,035명이 나옵니다
# -  총 발화(utterance) 수는 304,713번입니다
#
# 이 데이터셋은 규모도 크고 내용도 다양하며, 격식체와 비격식체, 여러
# 시간대, 여러 감정 상태 등이 두루 포함되어 있습니다. 우리의 바람은
# 이러한 다양성으로 인해 모델이 견고해지는, 즉 모델이 여러 종류의 입력
# 및 질의에 잘 대응할 수 있게 되는 것입니다.
#
# 우선은 원본 데이터 파일을 몇 줄 살펴보면서 형식이 어떻게 되어있는지
# 살펴 보겠습니다.
#

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "movie_lines.txt"))


######################################################################
# 원하는 형식의 데이터 파일로 만들기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 편의를 위해 데이터의 형식을 원하는 형태로 만들려고 합니다. 각 줄에
# *질의 문장* 과 *응답 문장* 의 쌍이 탭으로 구분되어 있게끔 하는 것입니다.
#
# 다음의 함수를 통해 *movie_lines.txt* 원본 데이터 파일을 파싱하려
# 합니다.
#
# -  ``loadLines`` 는 파일에 포함된 대사를 변환하여 항목(대사 ID ``lineID``,
#    인물 ID ``characterID``, 영화 ID ``movieID``, 인물 ``character``, 대사
#    내용 ``text``)에 대한 사전 형태로 변환합니다
# -  ``loadConversations`` 는 ``loadLines`` 를 통해 읽어들인
#    대사(``lines``)의 항목(``fields``)를 *movie_conversations.txt* 에 나와
#    있는 내용에 맞춰 대화 형태로 묶습니다
# -  ``extractSentencePairs`` 는 대화(``conversations``)에서 문장 쌍을
#    추출합니다
#

# 파일에 포함된 대사를 쪼개서 항목에 대한 사전(``dict``) 형태로 변환합니다
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # 항목을 추출합니다
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# 대사의 항목을 *movie_conversations.txt* 를 참고하여 대화 형태로 묶습니다
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # 항목을 추출합니다
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # 문자열을 리스트로 변환합니다(convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            # 대사를 재구성합니다
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# conversations에서 문장 쌍을 추출합니다
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # 대화를 이루는 각 대사에 대해 반복문을 수행합니다
        # 대화의 마지막 대사는 (그에 대한 응답이 없으므로) 무시합니다
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # 잘못된 샘플은 제거합니다(리스트가 하나라도 비어 있는 경우)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


######################################################################
# 이제 이 함수들을 호출하여 새로운 파일인 *formatted_movie_lines.txt* 를
# 만듭니다.
#

# 새 파일에 대한 경로를 정의합니다
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# 구분자에 대해 unescape 함수를 호출합니다
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# 대사 사전(dict), 대화 리스트(list), 그리고 각 항목의 이름을 초기화합니다
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# 대사(lines)를 읽어들여 대화(conversations)로 재구성합니다
print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

# 결과를 새로운 csv 파일로 저장합니다
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# 몇 줄을 예제 삼아 출력해 봅니다
print("\nSample lines from file:")
printLines(datafile)


######################################################################
# 데이터 읽고 정리하기
# ~~~~~~~~~~~~~~~~~~~~
#
# 다음에 해야 할 일은 어휘집을 만들고, 질의/응답 문장 쌍을 메모리로
# 읽어들이는 것입니다.
#
# 우리가 다루는 대상은 일련의 **단어** 들이며, 따라서 이들을 이산 공간 상의
# 수치(discrete numerical space)로 자연스럽게 대응시키기 어렵다는 점에
# 유의하시기 바랍니다. 따라서 우리는 데이터셋 안에 들어 있는 단어를 인덱스
# 값으로 변환하는 매핑을 따로 만들어야 합니다.
#
# 이를 위해 우리는 ``Voc`` 라는 클래스를 만들어 단어에서 인덱스로의
# 매핑, 인덱스에서 단어로의 역 매핑, 각 단어의 등장 횟수, 전체 단어 수
# 등을 관리하려 합니다. 이 클래스는 어휘집에 새로운 단어를 추가하는
# 메서드(``addWord``), 문장에 등장하는 모든 단어를 추가하는
# 메서드(``addSentence``), 그리고 자주 등장하지 않는 단어를 정리하는
# 메서드(``trim``)를 제공합니다. 단어를 정리하는 내용에 대해서는 뒤에서
# 좀 더 자세히 살펴보겠습니다.
#

# 기본 단어 토큰 값
PAD_token = 0  # 짧은 문장을 채울(패딩, PADding) 때 사용할 제로 토큰
SOS_token = 1  # 문장의 시작(SOS, Start Of Sentence)을 나타내는 토큰
EOS_token = 2  # 문장의 끝(EOS, End Of Sentence)을 나태는 토큰

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # SOS, EOS, PAD를 센 것

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 등장 횟수가 기준 이하인 단어를 정리합니다
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # 사전을 다시 초기화힙니다
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # 기본 토큰을 센 것

        for word in keep_words:
            self.addWord(word)


######################################################################
# 이제 어휘집과 질의/응답 문장 쌍을 재구성하려 합니다. 그러한 데이터를
# 사용하려면 그 전에 약간의 전처리 작업을 수행해야 합니다.
#
# 우선, ``unicodeToAscii`` 를 이용하여 유니코드 문자열을 아스키로 변환해야
# 합니다. 다음에는 모든 글자를 소문자로 변환하고, 알파벳도 아니고 기본적인
# 문장 부호도 아닌 글자는 제거합니다(정규화, ``normalizeString``).
# 마지막으로는 학습할 때의 편의성을 위해서, 길이가 일정 기준을 초과하는,
# 즉 ``MAX_LENGTH`` 보다 긴 문장을 제거합니다(``filterPairs``).
#

MAX_LENGTH = 10  # 고려할 문장의 최대 길이

# 유니코드 문자열을 아스키로 변환합니다
# https://stackoverflow.com/a/518232/2809427 참고
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 소문자로 만들고, 공백을 넣고, 알파벳 외의 글자를 제거합니다
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# 질의/응답 쌍을 읽어서 voc 객체를 반환합니다
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # 파일을 읽고, 쪼개어 lines에 저장합니다
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # 각 줄을 쪼개어 pairs에 저장하고 정규화합니다
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# 문장의 쌍 'p'에 포함된 두 문장이 모두 MAX_LENGTH라는 기준보다 짧은지를 반환합니다
def filterPair(p):
    # EOS 토큰을 위해 입력 시퀀스의 마지막 단어를 보존해야 합니다
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# 조건식 filterPair에 따라 pairs를 필터링합니다
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# 앞에서 정의한 함수를 이용하여 만든 voc 객체와 리스트 pairs를 반환합니다
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# voc와 pairs를 읽고 재구성합니다
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# 검증을 위해 pairs의 일부 내용을 출력해 봅니다
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


######################################################################
# 학습 단계가 빨리 수렴하도록 하는 또 다른 전략은 자주 쓰이지 않는 단어를
# 어휘집에서 제거하는 것입니다. 피처 공간의 크기를 줄이면 모델이
# 학습을 통해 근사하려는 함수의 난이도를 낮추는 효과도 있습니다. 우리는
# 이를 두 단계로 나눠 진행하려 합니다.
#
# 1) ``voc.trim`` 함수를 이용하여 ``MIN_COUNT`` 라는 기준 이하의 단어를
#    제거합니다.
#
# 2) 제거하기로 한 단어를 포함하는 경우를 pairs에서 제외합니다.
#

MIN_COUNT = 3    # 제외할 단어의 기준이 되는 등장 횟수

def trimRareWords(voc, pairs, MIN_COUNT):
    # MIN_COUNT 미만으로 사용된 단어는 voc에서 제외합니다
    voc.trim(MIN_COUNT)
    # 제외할 단어가 포함된 경우를 pairs에서도 제외합니다
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # 입력 문장을 검사합니다
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 출력 문장을 검사합니다
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # 입출력 문장에 제외하기로 한 단어를 포함하지 않는 경우만을 남겨둡니다
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# voc와 pairs를 정돈합니다
pairs = trimRareWords(voc, pairs, MIN_COUNT)


######################################################################
# 모델을 위한 데이터 준비하기
# ---------------------------
#
# 상당한 노력을 기울여 데이터를 전처리하고, 잘 정리하여 어휘집 객체와
# 문장 쌍의 리스트 형태로 만들어두긴 했지만, 결국 우리가 만들 모델에서
# 사용하는 입력은 수치 값으로 이루어진 torch 텐서입니다. 처리한 데이터를
# 모델에 맞는 형태로 준비하는 방법의 하나가 `seq2seq 변환 튜토리얼
# <https://tutorials.pytorch.kr/intermediate/seq2seq_translation_tutorial.html>`__
# 에 나와 있습니다. 이 튜토리얼에서는 배치 크기로 1을 사용하며, 이는 즉
# 문장에 등장하는 단어를 어휘집에서의 인덱스로 변환하여 모델에 제공하기만
# 하면 된다는 의미입니다.
#
# 그래도 여러분이 학습 속도나 GPU 병렬 처리 용량을 향상하고 싶다면
# 미니배치를 이용하여 학습해야 할 것입니다.
#
# 미니배치를 사용한다는 것은 배치에 포함된 문장 길이가 달라질 수 있다는
# 점에 유의해야 한다는 것을 뜻합니다. 같은 배치 안에서 크기가 다른
# 문장을 처리하기 위해서는 배치용 입력 텐서의 모양을 *(max_length,
# batch_size)* 로 맞춰야 합니다. 이때 *max_length* 보다 짧은 문장에
# 대해서는 *EOS 토큰* 뒤에 제로 토큰을 덧붙이면 됩니다.
#
# 영어로 된 문장을 텐서로 변환하기 위해 단순히 그에 대응하는 인덱스를
# 사용하고(``indexesFromSentence``) 제로 토큰을 패딩한다고 해봅시다.
# 그러면 텐서의 모양이 *(batch_size, max_length)* 이 되고, 첫 번째 차원에
# 대해 인덱싱을 수행하면 모든 시간대별 문장이 전부 반환될 것입니다.
# 그러나 우리는 배치를 시간에 따라, 그리고 배치에 포함된 모든 문장에
# 대해 인덱싱할 수도 있어야 합니다. 따라서 우리는 입력 배치의 모양을
# 뒤집어서 *(max_length, batch_size)* 형태로 만들 것입니다. 그러고 난
# 후에 첫 번째 차원에 대해 인덱싱하면 배치에 포함된 모든 문장을 시간에
# 대해 인덱싱한 결과를 반환하게 됩니다. 우리는 이 뒤집기 작업을
# ``zeroPadding`` 함수를 이용하여 묵시적으로 수행할 것입니다.
#
# .. figure:: /_static/img/chatbot/seq2seq_batches.png
#    :align: center
#    :alt: batches
#
# ``inputVar`` 함수는 문장을 텐서로 변환하는, 그리고 궁극적으로는 제로
# 패딩하여 올바른 모양으로 맞춘 텐서를 만드는 작업을 수행합니다. 이
# 함수는 각 배치에 포함된 시퀀스의 길이(``lengths``)로 구성된 텐서도 같이
# 반환합니다. 그리고 우리는 이를 나중에 디코더로 넘겨줄 것입니다.
#
# ``outputVar`` 함수는 ``inputVar`` 와 비슷한 작업을 수행하지만, ``lengths``
# 텐서를 반환하는 대신에 이진 마스크로 구성된 텐서와 목표 문장의 최대
# 길이를 같이 반환합니다. 이진 마스크 텐서는 출력에 해당하는 목표 텐서와
# 그 모양이 같지만, 패딩 토큰(*PAD_token*)에 해당하는 경우에는 값이 0이며
# 나머지 경우의 값은 1입니다.
#
# ``batch2TrainData`` 는 단순히 여러 쌍을 입력으로 받아서, 앞서 설명한
# 함수를 이용하여 입력 및 목표 텐서를 구하여 반환합니다.
#

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# 입력 시퀀스 텐서에 패딩한 결과와 lengths를 반환합니다
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# 패딩한 목표 시퀀스 텐서, 패딩 마스크, 그리고 최대 목표 길이를 반환합니다
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# 입력 배치를 이루는 쌍에 대한 모든 아이템을 반환합니다
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# 검증용 예시
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


######################################################################
# 모델 정의하기
# -------------
#
# Seq2Seq 모델
# ~~~~~~~~~~~~
#
# 우리 챗봇의 두뇌에 해당하는 모델은 sequence-to-sequence (seq2seq)
# 모델입니다. seq2seq 모델의 목표는 가변 길이 시퀀스를 입력으로 받고,
# 크기가 고정된 모델을 이용하여, 가변 길이 시퀀스를 출력으로 반환하는
# 것입니다.
#
# `Sutskever 등 <https://arxiv.org/abs/1409.3215>`__ 은 두 개의 독립된
# 순환 신경망을 같이 이용하여 이러한 목적을 달성할 수 있음을 발견했습니다.
# RNN 하나는 **인코더** 로, 가변 길이 입력 시퀀스를 고정된 길이의 문맥
# 벡터(context vector)로 인코딩합니다. 이론상 문맥 벡터(RNN의 마지막
# 은닉 레이어)는 봇에게 입력으로 주어지는 질의 문장에 대한 의미론적 정보를
# 담고 있을 것입니다. 두 번째 RNN은 **디코더** 입니다. 디코더는 단어 하나와
# 문맥 벡터를 입력으로 받고, 시퀀스의 다음 단어가 무엇일지를 추론하여
# 반환하며, 다음 단계에서 사용할 은닉 상태도 같이 반환합니다.
#
# .. figure:: /_static/img/chatbot/seq2seq_ts.png
#    :align: center
#    :alt: model
#
# 그림 출처:
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
#


######################################################################
# 인코더
# ~~~~~~
#
# 인코더 RNN은 입력 시퀀스를 토큰 단위로(예를 들어, 단어 단위로) 한번에
# 하나씩 살펴보며 진행합니다. 그리고 각 단계마다 "출력" 벡터와 "은닉
# 상태" 벡터를 반환합니다. 은닉 상태 벡터는 다음 단계를 진행할 때 같이
# 사용되며, 출력 벡터는 차례대로 기록됩니다. 인코더는 시퀀스의 각 지점에
# 대해 파악한 문맥을 고차원 공간에 있는 점들의 집합으로 변환합니다.
# 나중에 디코더는 이를 이용하여 주어진 문제에 대해 의미 있는 출력을
# 구할 것입니다.
#
# 인코더의 핵심 부분에는 다중 레이어 게이트 순환 유닛(multi-layered Gated
# Recurrent Unit)이 있습니다. 이는 `Cho 등 <https://arxiv.org/pdf/1406.1078v3.pdf>`__
# 이 2014년에 고안한 것입니다. 우리는 GRU를 양방향으로 변환한 형태를
# 사용할 것이며, 이는 본질적으로 두 개의 독립된 RNN이 존재한다는
# 의미입니다. 하나는 입력 시퀀스를 원래 시퀀스에서의 순서로 처리하며,
# 다른 하나는 입력 시퀀스를 역순으로 처리합니다. 단계마다 각 네트워크의
# 출력을 합산합니다. 양방향 GRU를 사용하면 과거와 미래의 문맥을 함께
# 인코딩할 수 있다는 장점이 있습니다.
#
# 양방향 RNN:
#
# .. figure:: /_static/img/chatbot/RNN-bidirectional.png
#    :width: 70%
#    :align: center
#    :alt: rnn_bidir
#
# 그림 출처: https://colah.github.io/posts/2015-09-NN-Types-FP/
#
# ``embedding`` 레이어가 단어 인덱스를 임의 크기의 피처 공간으로
# 인코딩하는 데 사용되었음에 유의하기 바랍니다. 우리의 모델에서는 이
# 레이어가 각 단어를 크기가 *hidden_size* 인 피처 공간으로 매핑할
# 것입니다. 학습을 거치면 서로 뜻이 유사한 단어는 의미적으로 유사하게
# 인코딩될 것입니다.
#
# 마지막으로, RNN 모듈에 패딩된 배치를 보내려면 RNN과 연결된 부분에서
# 패킹 및 언패킹하는 작업을 수행해야 합니다. 각각은
# ``nn.utils.rnn.pack_padded_sequence`` 와
# ``nn.utils.rnn.pad_packed_sequence`` 를 통해 수행할 수 있습니다.
#
# **계산 그래프:**
#
#    1) 단어 인덱스를 임베딩으로 변환합니다.
#    2) RNN 모듈을 위한 패딩된 배치 시퀀스를 패킹합니다.
#    3) GRU로 포워드 패스를 수행합니다.
#    4) 패딩을 언패킹합니다.
#    5) 양방향 GRU의 출력을 합산합니다.
#    6) 출력과 마지막 은닉 상태를 반환합니다.
#
# **입력:**
#
# -  ``input_seq``: 입력 시퀀스 배치. shape=\ *(max_length,
#    batch_size)*
# -  ``input_lengths``: 배치에 포함된 각 문장의 길이로 구성된 리스트.
#    shape=\ *(batch_size)*
# -  ``hidden``: 은닉 상태. shape=\ *(n_layers x num_directions,
#    batch_size, hidden_size)*
#
# **출력:**
#
# -  ``outputs``: GRU의 마지막 은닉 레이어에 대한 출력 피처 값(양방향
#    (출력을 합산한 것). shape=\ *(max_length, batch_size, hidden_size)*
# -  ``hidden``: GRU의 최종 은닉 상태. shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
#
#

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # GRU를 초기화합니다. input_size와 hidden_size 매개변수는 둘 다 'hidden_size'로
        # 둡니다. 이는 우리 입력의 크기가 hideen_size 만큼의 피처를 갖는 단어 임베딩이기
        # 때문입니다.
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 단어 인덱스를 임베딩으로 변환합니다
        embedded = self.embedding(input_seq)
        # RNN 모듈을 위한 패딩된 배치 시퀀스를 패킹합니다
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # GRU로 포워드 패스를 수행합니다
        outputs, hidden = self.gru(packed, hidden)
        # 패딩을 언패킹합니다
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # 양방향 GRU의 출력을 합산합니다
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # 출력과 마지막 은닉 상태를 반환합니다
        return outputs, hidden


######################################################################
# 디코더
# ~~~~~~
#
# 디코더 RNN은 토큰 단위로 응답 문장을 생성하는 역할을 수행합니다. 이때
# 인코더의 문맥 벡터를 사용하며, 내부 은닉 상태에 따라 시퀀스의 다음
# 단어를 생성하게 됩니다. 디코더는 *EOS_token*, 즉 문장의 끝을 나타내는
# 토큰을 출력할 때까지 계속 단어를 생성합니다. 원래의 seq2seq 디코더에는
# 알려진 문제점이 있습니다. 만약 우리가 입력 시퀀스의 의미를 인코딩할
# 때 문맥 벡터에만 전적으로 의존한다면, 그 과정 중에 정보 손실이 일어날
# 가능성이 높다는 것입니다. 이는 특히 입력 시퀀스의 길이가 길 때 그러하며,
# 이 때문에 디코더의 기능이 크게 제한될 수 있습니다.
#
# 이를 해결하기 위한 방편으로, `Bahdanau 등
# <https://arxiv.org/abs/1409.0473>`__ 은 '어텐션 메커니즘'을
# 고안했습니다. 이는 디코더가 매 단계에 대해 고정된 문맥을 계속 사용하는
# 것이 아니라, 입력 시퀀스의 특정 부분에 집중하게 하는 방식입니다.
#
# 높은 차원에서 이야기 하자면, 어텐션은 디코더의 현재 은닉 상태와 인코더의
# 출력을 바탕으로 계산됩니다. 출력되는 어텐션 가중치는 입력 시퀀스와
# 동일한 모양을 가집니다. 따라서 이를 인코더의 출력과 곱할 수 있고, 그
# 결과로 얻게 되는 가중치 합은 인코더의 출력에서 어느 부분에 집중해야
# 할지를 알려줍니다. `Sean Robertson <https://github.com/spro>`__
# 의 그림에 이러한 내용이 잘 설명되어 있습니다.
#
# .. figure:: /_static/img/chatbot/attn2.png
#    :align: center
#    :alt: attn2
#
# `Luong 등 <https://arxiv.org/abs/1508.04025>`__ 은 Bahdanau의 기초 연구를
# 더욱 발전시킨 '전역(global) 어텐션'을 제안했습니다. '전역 어텐션'의
# 핵심적인 차이점은 인코더의 은닉 상태를 모두 고려한다는 점입니다. 이는
# Bahdanau 등의 '지역(local) 어텐션' 방식이 현재 시점에 대한 인코더의
# 은닉 상태만을 고려한다는 점과 다른 부분입니다. '전역 어텐션'의 또 다른
# 차이점은 어텐션에 대한 가중치, 혹은 에너지를 계산할 때 현재 시점에 대한
# 디코더의 은닉 상태만을 사용한다는 점입니다. Bahdanau 등은 어텐션을
# 계산할 때 디코더의 이전 단계 상태에 대한 정보를 활용합니다. 또한 Luong 등의
# 방법에서는 인코더의 출력과 디코더의 출력에 대한 어텐션 에너지를 계산하는
# 방법을 제공하며, 이를 '점수 함수(score function)'라 부릅니다.
#
# .. figure:: /_static/img/chatbot/scores.png
#    :width: 60%
#    :align: center
#    :alt: scores
#
# 이때 :math:`h_t` 는 목표 디코더의 현재 상태를, :math:`\bar{h}_s` 는 인코더의
# 모든 상태를 뜻합니다.
#
# 종합해 보면, 전역 어텐션 메커니즘을 다음 그림과 같이 요약할 수 있을
# 것입니다. 우리가 '어텐션 레이어'를 ``Attn`` 라는 독립적인 ``nn.Module`` 로
# 구현할 것임에 유의하기 바랍니다. 이 모듈의 출력은 모양이 *(batch_size, 1,
# max_length)* 인 정규화된 softmax 가중치 텐서입니다.
#
# .. figure:: /_static/img/chatbot/global_attn.png
#    :align: center
#    :width: 60%
#    :alt: global_attn
#

# Luong 어텐션 레이어
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Attention 가중치(에너지)를 제안된 방법에 따라 계산합니다
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # max_length와 batch_size의 차원을 뒤집습니다
        attn_energies = attn_energies.t()

        # 정규화된 softmax 확률 점수를 반환합니다 (차원을 늘려서)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


######################################################################
# 이처럼 어텐션 서브모듈을 정의하고 나면 실제 디코더 모델을 구현할 수
# 있게 됩니다. 디코더에 대해서는 매 시간마다 배치를 하나씩 수동으로
# 제공하려 합니다. 이는 임베딩된 단어 텐서와 GRU 출력의 모양이 둘 다
# *(1, batch_size, hidden_size)* 라는 의미입니다.
#
# **계산 그래프:**
#
#    1) 현재의 입력 단어에 대한 임베딩을 구합니다.
#    2) 무방향 GRU로 포워드 패스를 수행합니다.
#    3) (2)에서 구한 현재의 GRU 출력을 바탕으로 어텐션 가중치를 계산합니다.
#    4) 인코더 출력에 어텐션을 곱하여 새로운 "가중치 합" 문맥 벡터를 구합니다.
#    5) Luong의 논문에 나온 식 5를 이용하여 가중치 문맥 벡터와 GRU 출력을 결합합니다.
#    6) Luong의 논문에 나온 식 6을 이용하여(softmax 없이) 다음 단어를 예측합니다.
#    7) 출력과 마지막 은닉 상태를 반환합니다.
#
# **입력:**
#
# -  ``input_step``: 입력 시퀀스 배치에 대한 한 단위 시간(한 단어).
#    shape=\ *(1, batch_size)*
# -  ``last_hidden``: GRU의 마지막 은닉 레이어. shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
# -  ``encoder_outputs``: 인코더 모델의 출력. shape=\ *(max_length,
#    batch_size, hidden_size)*
#
# **출력:**
#
# -  ``output``: 각 단어가 디코딩된 시퀀스에서 다음 단어로 사용되었을
#    때 적합할 확률을 나타내는 정규화된 softmax 텐서.
#    shape=\ *(batch_size, voc.num_words)*
# -  ``hidden``: GRU의 마지막 은닉 상태. shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
#

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # 참조를 보존해 둡니다
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 레이어를 정의합니다
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 주의: 한 단위 시간에 대해 한 단계(단어)만을 수행합니다
        # 현재의 입력 단어에 대한 임베딩을 구합니다
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # 무방향 GRU로 포워드 패스를 수행합니다
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 현재의 GRU 출력을 바탕으로 어텐션 가중치를 계산합니다
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # 인코더 출력에 어텐션을 곱하여 새로운 "가중치 합" 문맥 벡터를 구합니다
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Luong의 논문에 나온 식 5를 이용하여 가중치 문맥 벡터와 GRU 출력을 결합합니다
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Luong의 논문에 나온 식 6을 이용하여 다음 단어를 예측합니다
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # 출력과 마지막 은닉 상태를 반환합니다
        return output, hidden


######################################################################
# 학습 프로시저 정의하기
# ----------------------
#
# Masked loss
# ~~~~~~~~~~~
#
# 우리는 패딩된 시퀀스 배치를 다루기 때문에 손실을 계산할 때 단순히 텐서의
# 모든 원소를 고려할 수는 없습니다. 우리는 ``maskNLLLoss`` 를 정의하여
# 디코더의 출력 텐서, 목표 텐서, 이진 마스크 텐서를 바탕으로 손실을 계산하려
# 합니다. 이 손실 함수에서는 마스크 텐서의 *1* 에 대응하는 원소에 대한 음의
# 로그 우도 값의 평균을 계산합니다.
#

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


######################################################################
# 한 번의 학습 단계
# ~~~~~~~~~~~~~~~~~
#
# ``train`` 함수에 학습을 한 단계(입력 배치 한 개에 대한) 진행하는 알고리즘이
# 나와 있습니다.
#
# 우리는 수렴이 잘 되도록 몇 가지 영리한 전략을 사용해보려 합니다.
#
# -  첫 번째 전략은 **teacher forcing** 을 사용하는 것입니다. 이는
#    ``teacher_forcing_ratio`` 로 정의된 확률에 따라, 디코더의 이번 단계
#    예측값 대신에 현재의 목표 단어를 디코더의 다음 입력 값으로 활용하는
#    것입니다. 이 기법은 디코더의 보조 바퀴처럼 작용하여 효율적으로 학습될 수
#    있게 도와 줍니다. 하지만 teacher forcing 기법은 추론 과정에서 모델이
#    불안정 해지도록 할 수도 있는데, 이는 디코더가 학습 과정에서 자신의 출력
#    시퀀스를 직접 만들어 볼 기회를 충분히 제공받지 못할 수 있기 때문입니다.
#    따라서 우리는 ``teacher_forcing_ratio`` 를 어떻게 설정해 두었는지에
#    주의를 기울여야 하며, 수렴이 빨리 되었다고 속아 넘어가서는 안 됩니다.
#
# -  우리가 구현한 두 번째 전략은 **gradient clipping** 입니다. 이는 소위
#    '그라디언트 폭발' 문제를 해결하기 위해 널리 사용되는 기법입니다. 핵심은
#    그라디언트를 클리핑 하거나 임계값을 둠으로써, 그라디언트가 지수
#    함수적으로 증가하거나 오버플로를 일으키는(NaN) 경우를 막고, 비용 함수의
#    급격한 경사를 피하겠다는 것입니다.
#
# .. figure:: /_static/img/chatbot/grad_clip.png
#    :align: center
#    :width: 60%
#    :alt: grad_clip
#
# 그림 출처: Goodfellow 등 저. *Deep Learning*. 2016. https://www.deeplearningbook.org/
#
# **작업 절차:**
#
#    1) 전체 입력 배치에 대하여 인코더로 포워드 패스를 수행합니다.
#    2) 디코더의 입력을 SOS_token로, 은닉 상태를 인코더의 마지막 은닉 상태로 초기화합니다.
#    3) 입력 배치 시퀀스를 한 번에 하나씩 디코더로 포워드 패스합니다.
#    4) Teacher forcing을 사용하는 경우, 디코더의 다음 입력을 현재의 목표로 둡니다. 그렇지 않으면 디코더의 다음 입력을 현재 디코더의 출력으로 둡니다.
#    5) 손실을 계산하고 누적합니다.
#    6) 역전파를 수행합니다.
#    7) 그라디언트를 클리핑 합니다.
#    8) 인코더 및 디코더 모델의 매개변수를 갱신합니다.
#
#
# .. warning::
#
#   PyTorch의 RNN 모듈(``RNN``, ``LSTM``, ``GRU``)은 전체 입력 시퀀스(또는
#   시퀀스의 배치)를 단순히 넣어주기만 하면 다른 비순환 레이어처럼 사용할 수
#   있습니다. 우리는 ``encoder`` 에서 ``GRU`` 레이어를 이런 식으로 사용합니다.
#   그 안이 실제로 어떻게 되어 있는지를 살펴보면, 매 시간 단계마다 은닉 상태를
#   계산하는 반복 프로세스가 존재합니다. 또 다른 방법은, 이 모듈을 매번 한 단위
#   시간만큼 수행할 수도 있습니다. 그 경우에는 우리가 ``decoder`` 모델을 다룰
#   때처럼, 학습 과정에서 수동으로 시퀀스에 대해 반복 작업을 수행해 주어야
#   합니다. 이 모듈에 대해 모델의 개념을 확실히 갖고만 있다면, 순차 모델을
#   구현하는 것도 매우 단순할 것입니다.
#
#


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # 제로 그라디언트
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # device 옵션을 설정합니다
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # 변수를 초기화합니다
    loss = 0
    print_losses = []
    n_totals = 0

    # 인코더로 포워드 패스를 수행합니다
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # 초기 디코더 입력을 생성합니다(각 문장을 SOS 토큰으로 시작합니다)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # 디코더의 초기 은닉 상태를 인코더의 마지막 은닉 상태로 둡니다
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 이번 반복에서 teacher forcing을 사용할지를 결정합니다
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 배치 시퀀스를 한 번에 하나씩 디코더로 포워드 패스합니다
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing 사용: 다음 입력을 현재의 목표로 둡니다
            decoder_input = target_variable[t].view(1, -1)
            # 손실을 계산하고 누적합니다
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing 미사용: 다음 입력을 디코더의 출력으로 둡니다
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # 손실을 계산하고 누적합니다
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 역전파를 수행합니다
    loss.backward()

    # 그라디언트 클리핑: 그라디언트를 제자리에서 수정합니다
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 모델의 가중치를 수정합니다
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


######################################################################
# 학습 단계
# ~~~~~~~~~
#
# 이제 마지막으로 전체 학습 프로시저와 데이터를 하나로 엮을 때가
# 되었습니다. ``trainIters`` 함수는 주어진 모델, optimizer, 데이터 등을
# 토대로 학습을 ``n_iterations`` 번의 단계만큼 진행하는 역할을 담당합니다.
# 이 함수는 자기 자신을 살 설명하고 있는 편인데, 무거운 작업을 ``train``
# 함수에 옮겨 놓았기 때문입니다.
#
# 한 가지 주의할 점은 우리가 모델을 저장하려 할 때, 인코더와 디코더의
# state_dicts (매개변수), optimizer의 state_dicts, 손실, 진행 단계 수
# 등을 tarball로 만들어 저장한다는 점입니다. 모델을 이러한 방식으로
# 저장하면 checkpoint에 대해 아주 높은 수준의 유연성을 확보할 수 있게
# 됩니다. Checkpoint를 불러오고 나면, 우리는 모델 매개변수를 이용하여
# 예측을 진행할 수도 있고, 이전에 멈췄던 부분부터 학습을 계속  진행할
# 수도 있게 됩니다.
#

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # 각 단계에 대한 배치를 읽어옵니다
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # 초기화
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # 학습 루프
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # 배치에서 각 필드를 읽어옵니다
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # 배치에 대해 학습을 한 단계 진행합니다
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # 경과를 출력합니다
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Checkpoint를 저장합니다
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


######################################################################
# 평가 정의하기
# -------------
#
# 모델을 학습시키고 나면 직접 봇과 대화를 나눠보고 싶어질 것입니다. 그러려면
# 먼저 모델이 인코딩된 입력을 어떻게 디코딩할지를 정의해줘야 합니다.
#
# 탐욕적 디코딩
# ~~~~~~~~~~~~~
#
# 탐욕적 디코딩(Greedy decoding)은 우리가 학습 단계에서 teacher forcing을
# 적용하지 않았을 때 사용한 디코딩 방법입니다. 달리 말하면, 각 단계에 대해
# 단순히 ``decoder_output`` 에서 가장 높은 softmax값을 갖는 단어를 선택하는
# 방식입니다. 이 디코딩 방법은 한 번의 단계에 대해서는 최적입니다.
#
# 우리는 탐욕적 디코딩 연산을 수행할 수 있도록 ``GreedySearchDecoder``
# 클래스를 만들었습니다. 수행 과정에서 이 클래스의 인스턴스는 모양이
# *(input_seq length, 1)* 인 입력 시퀀스(``input_seq``), 조종할 입력
# 길이(``input_length``) 텐서, 그리고 응답 문장 길이의 제한을 나타내는
# ``max_length`` 를 입력으로 받습니다. 입력 시퀀서는 다음과 같은 계산 그래프에
# 의해 평가됩니다.
#
# **계산 그래프:**
#
#    1) 인코더 모델로 입력을 포워드 패스합니다.
#    2) 인코더의 마지막 은닉 레이어가 디코더의 첫 번째 은닉 레이어의 입력이 되도록 준비합니다.
#    3) 디코더의 첫 번째 입력을 SOS_token으로 초기화합니다.
#    4) 디코더가 단어를 덧붙여 나갈 텐서를 초기화합니다.
#    5) 반복적으로 각 단계마다 하나의 단어 토큰을 디코딩합니다.
#        a) 디코더로의 포워드 패스를 수행합니다.
#        b) 가장 가능성 높은 단어 토큰과 그 softmax 점수를 구합니다.
#        c) 토큰과 점수를 기록합니다.
#        d) 현재의 토큰을 디코더의 다음 입력으로 준비시킵니다.
#    6) 단어 토큰과 점수를 모아서 반환합니다.
#

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # 인코더 모델로 입력을 포워드 패스합니다
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 인코더의 마지막 은닉 레이어가 디코더의 첫 번째 은닉 레이어의 입력이 되도록 준비합니다
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # 디코더의 첫 번째 입력을 SOS_token으로 초기화합니다
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # 디코더가 단어를 덧붙여 나갈 텐서를 초기화합니다
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # 반복적으로 각 단계마다 하나의 단어 토큰을 디코딩합니다
        for _ in range(max_length):
            # 디코더로의 포워드 패스를 수행합니다
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 가장 가능성 높은 단어 토큰과 그 softmax 점수를 구합니다
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 토큰과 점수를 기록합니다
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 현재의 토큰을 디코더의 다음 입력으로 준비시킵니다(차원을 증가시켜서)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 단어 토큰과 점수를 모아서 반환합니다
        return all_tokens, all_scores


######################################################################
# 내 텍스트 평가하기
# ~~~~~~~~~~~~~~~~~~
#
# 이제 디코딩 모델을 정의했으니, 문자열로 된 입력 시퀀스를 평가하는 함수를
# 작성해볼 수 있을 것입니다. ``evaluate`` 함수에 입력 시퀀스를 낮은
# 레벨에서 어떻게 처리할지가 나와 있습니다. 우리는 먼저 문장을
# *batch_size==1* 이고 단어 인덱스로 구성된 입력 배치 형태로 만듭니다.
# 이를 위해 문장의 각 단어를 그에 대응하는 인덱스로 변환하고, 차원을
# 뒤집어서 모델에 맞는 입력 형태로 변환합니다. 우리는 입력 시퀀스의 길이를
# 저장하고 있는 ``lengths`` 텐서도 만듭니다. 이 경우에는 ``lengths`` 가
# 스칼라 값이 되는데, 우리는 한 번에 한 문장만 평가하기 때문입니다(batch_size==1).
# 다음으로는 ``GreedySearchDecoder`` 의 객체(``searcher``)를 이용하여
# 응답 문장 텐서를 디코딩합니다. 마지막으로, 응답 인덱스를 단어로 변환하고
# 디코딩된 단어의 리스트를 반환합니다.
#
# ``evaluateInput`` 은 우리의 챗봇에 대한 인터페이스 역할을 수행합니다.
# 이를 호출하면 입력 텍스트 필드가 생성되는데, 거기에 우리의 질의 문장을
# 입력해볼 수 있습니다. 입력 문장을 타이핑하고 *엔터* 를 누르면, 입력한
# 텍스트가 학습 데이터와 같은 방식으로 정규화되고, 최종적으로는 ``evaluate``
# 함수에 입력으로 제공되어 디코딩된 출력 문장을 구하게 됩니다. 우리는
# 이러한 과정을 계속 반복하며, 이를 통해 'q'나 'quit'를 입력하기 전까지는
# 계속 채팅할 수 있습니다.
#
# 마지막으로, 만약 어휘집에 포함되어 있지 않은 단어를 포함하고 있는 문장이
# 입력되더라도 이를 예의 바르게 처리합니다. 즉 에러 메시지를 출력하고
# 사용자에게 새로운 문장을 입력해달라고 요청합니다.
#

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### 입력 시퀀스를 배치 형태로 만듭니다
    # 단어 -> 인덱스
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # lengths 텐서를 만듭니다
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 배치의 차원을 뒤집어서 모델이 사용하는 형태로 만듭니다
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 적절한 디바이스를 사용합니다
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # searcher를 이용하여 문장을 디코딩합니다
    tokens, scores = searcher(input_batch, lengths, max_length)
    # 인덱스 -> 단어
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 입력 문장을 받아옵니다
            input_sentence = input('> ')
            # 종료 조건인지 검사합니다
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 문장을 정규화합니다
            input_sentence = normalizeString(input_sentence)
            # 문장을 평가합니다
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 응답 문장을 형식에 맞춰 출력합니다
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


######################################################################
# 모델 수행하기
# -------------
#
# 마지막으로, 우리의 모델을 수행해 볼 시간입니다!
#
# 우리가 챗봇 모델을 학습할 때든 테스트할 때든, 우리는 각각의 인코더 및
# 디코더 모델을 초기화해줘야 합니다. 다음 블록에서는 우리가 원하는대로
# 설정을 맞추고, 처음부터 시작할지, 아니면 checkpoint를 불러올지 정하고,
# 모델을 빌드하고 초기화합니다. 성능을 최적화하기 위해서는 모델 설정을
# 여러가지로 바꿔 보면서 테스트해보기 바랍니다.
#

# 모델을 설정합니다
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# 불러올 checkpoint를 설정합니다. 처음부터 시작할 때는 None으로 둡니다.
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# loadFilename이 제공되는 경우에는 모델을 불러옵니다
if loadFilename:
    # 모델을 학습할 때와 같은 기기에서 불러오는 경우
    checkpoint = torch.load(loadFilename)
    # GPU에서 학습한 모델을 CPU로 불러오는 경우
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# 단어 임베딩을 초기화합니다
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# 인코더 및 디코더 모델을 초기화합니다
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# 적절한 디바이스를 사용합니다
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


######################################################################
# 학습 수행하기
# ~~~~~~~~~~~~~
#
# 모델을 학습해보고 싶다면 다음 블록을 수행하면 됩니다.
#
# 먼저 학습 매개변수를 설정하고, optimizer를 초기화한 뒤, 마지막으로 ``trainIters``
# 함수를 호출하여 학습 단계를 진행합니다.
#

# 학습 및 최적화 설정
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Dropout 레이어를 학습 모드로 둡니다
encoder.train()
decoder.train()

# Optimizer를 초기화합니다
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# cuda가 있다면 cuda를 설정합니다
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# 학습 단계를 수행합니다
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)


######################################################################
# 평가 수행하기
# ~~~~~~~~~~~~~
#
# 여러분의 모델과 채팅을 해보고 싶다면 다음 블록을 수행하면 됩니다.
#

# Dropout 레이어를 평가 모드로 설정합니다
encoder.eval()
decoder.eval()

# 탐색 모듈을 초기화합니다
searcher = GreedySearchDecoder(encoder, decoder)

# 채팅을 시작합니다 (다음 줄의 주석을 제거하면 시작해볼 수 있습니다)
# evaluateInput(encoder, decoder, searcher, voc)


######################################################################
# 맺음말
# ------
#
# 이번 튜토리얼을 이것으로 마무리하겠습니다. 축하합니다! 여러분은 이제 생성
# 챗봇 모델을 만들기 위한 기초 지식을 습득했습니다. 만약 좀 더 관심이 있다면
# 모델이나 학습 매개변수를 수정해 보면서, 혹은 모델을 학습할 데이터를 바꿔
# 보면서 챗봇의 행동을 수정해볼 수 있을 것입니다.
#
# 그 외에도 딥러닝의 멋진 활용 예에 대한 PyTorch 튜토리얼이 있으니 한 번
# 확인해 보기 바랍니다!
#
