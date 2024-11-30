# -*- coding: utf-8 -*-
"""
TorchText를 사용하여 사용자 정의 글 데이터셋 전처리하기
==========================================================

**번역**: `Anupam Sharma <https://anp-scp.github.io/>`_
**저자**: `장효영 <https://github.com/hyoyoung>`_

This tutorial illustrates the usage of torchtext on a dataset that is not built-in. In the tutorial,
we will preprocess a dataset that can be further utilized to train a sequence-to-sequence
model for machine translation (something like, in this tutorial: `Sequence to Sequence Learning
with Neural Networks <https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%\
20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb>`_) but without using legacy version
of torchtext.
이 튜토리얼에서는 기본 제공되지 않는 데이터셋에서 Torchtext를 사용하는 방법을 설명합니다.
튜토리얼 안에서는 기계 번역을 위한 시퀀스 간 모델을 훈련하는 데 추가로 활용할 수 있는 데이터셋를 전처리 할 것입니다.
(이 튜토리얼과 비슷합니다: `Sequence to Sequence Learning
with Neural Networks <https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%\
20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb>`_)
그러나 레거시 버전의 torchtext를 사용하지 않습니다.

이 튜토리얼에서는 아래 방법을 배워보겠습니다

* 데이터셋 읽기
* 문장 토큰화하기
* 문장에 변환 적용하기
* 버킷 배치 처리


영어에서 독일어 번역을 수행할 수 있는 모델을 훈련하기 위해 데이터셋을 준비해야 한다고 가정해 보겠습니다.
`Tatoeba Project <https://tatoeba.org/en>`_가 제공하는 탭으로 구분된 독일어-영어 문장 쌍을 사용하겠습니다.
이 데이터는 `다운로드 링크 <https://www.manythings.org/anki/deu-eng.zip>`__ 에서 받을 수 있습니다.

다른 언어에 대한 문장 쌍은 `다운로드 링크 <https://www.manythings.org/anki/>`\ 에서 찾을 수 있습니다.

__.
"""

# %%
# Setup
# -----
#
# 먼저 데이터셋을 다운로드하고, 압축을 푼 다음, `deu.txt` 파일의 경로를 적어둡니다.
#
# 다음 패키지가 설치되어 있는지 확인합니다
#
# * `Torchdata 0.6.0 <https://pytorch.org/data/beta/index.html>`_ (`설치 방법 \
#   <https://github.com/pytorch/data>`__)
# * `Torchtext 0.15.0 <https://pytorch.org/text/stable/index.html>`_ (`설치 방법 \
#   <https://github.com/pytorch/text>`__)
# * `Spacy <https://spacy.io/usage>`__
#
# 여기서는 `Spacy`를 사용하여 텍스트를 토큰화합니다. 간단히 말해서 토큰화는
# 문장을 단어의 리스트로 변환하는 것을 의미합니다. Spacy는 다양한 자연어 처리(NLP) 작업에 사용되는 파이썬 패키지입니다.
#
# 아래와 같이 Spacy에서 영어와 독일어 모델을 다운로드합니다
#
# .. code-block:: shell
#
#    python -m spacy download en_core_web_sm
#    python -m spacy download de_core_news_sm
#


# %%
# 필요한 모듈을 import 하면서 시작합니다

import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator
eng = spacy.load("en_core_web_sm") # 영어 모델을 로드하여 영어 텍스트를 토큰화합니다
de = spacy.load("de_core_news_sm") # 독일어 모델을 로드하여 독일어 텍스트를 토큰화합니다

# %%
# 이제 데이터셋을 읽어들입니다

FILE_PATH = 'data/deu.txt'
data_pipe = dp.iter.IterableWrapper([FILE_PATH])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)

# %%
# 위의 코드 블록에서는 다음과 같은 작업을 수행하고 있습니다
#
# 1. 2번째 줄에서 파일 이름의 반복가능한 객체를 생성하고 있습니다
# 2. 3번째 줄에서 해당 반복가능한 객체를 `FileOpener`에 전달하고,
#    파일을 읽기 모드로 열게 됩니다.
# 3. 4번째 줄에서 해당 파일을 파싱하는 함수를 호출합니다.
#    해당 함수는 탭으로 구분된 파일의 각각 줄(row)이 있는 반복 가능한 튜플 객체를 리턴합니다.
#
# DataPipe는 다양한 동작을 수행할수 있는 데이터 셋 객체와 비슷하게 생각할 수 있습니다.
# DataPipe에 관한 자세한 내용은 `해당 튜토리얼 <https://pytorch.org/data/beta/dp_tutorial.html>`_ 을 확인하세요.
#
# 반복가능한 객체가 아래와 같은 문장 쌍을 지녔는지 확인할 수 있습니다.

for sample in data_pipe:
    print(sample)
    break

# %%
# 한 쌍의 문장과 함께 속성 세부 사항이 같이 있다는 점을 눈여겨 보십시요.
# 속성 세부 정보를 제거할수 있는 작은 함수를 작성해봅시다.

def removeAttribution(row):
    """
    처음 두 요소를 튜플에 유지하는 함수
    """
    return row[:2]
data_pipe = data_pipe.map(removeAttribution)

# %%
# 위 코드 블록의 6번째 줄에 있는 `map` 함수는 `data_pipe`의 각 요소에 대해
# 어떠한 함수를 적용하는 데 사용할 수 있습니다. 이제 `data_pipe`에 다음과 같은 문장의
# 문장 쌍만 포함되어 있음을 확인할 수 있습니다.


for sample in data_pipe:
    print(sample)
    break

# %%
# 이제 토큰화를 수행하는 몇 가지 함수를 정의해 보겠습니다

def engTokenize(text):
    """
    영어 텍스트를 토큰화하여 토큰 리스트를 반환합니다.
    """
    return [token.text for token in eng.tokenizer(text)]

def deTokenize(text):
    """
    독일어 텍스트를 토큰화하고 토큰 리스트를 반환합니다
    """
    return [token.text for token in de.tokenizer(text)]

# %%
# 위 함수는 아래와 같이 텍스트를 받아 단어 리스트를 반환합니다

print(engTokenize("Have a good day!!!"))
print(deTokenize("Haben Sie einen guten Tag!!!"))

# %%
# 어휘 구축하기
# -----------------------
# Let us consider an English sentence as the source and a German sentence as the target.
# 영어 문장을 소스로, 독일어 문장을 타겟으로 생각해 봅시다
#
# 어휘는 데이터셋에 있는 고유한 단어의 집합으로 간주할 수 있습니다
# 이제 소스와 타겟 모두에 대한 어휘를 구축하겠습니다
#
# 반복자의 튜플 요소에서 토큰을 가져오는 함수를 정의해 보겠습니다


def getTokens(data_iter, place):
    """
    반복자에서 토큰을 생성(yield)하는 함수. 반복자는 문장의 튜플을 포함 (소스와 타겟).
    `place` 매개변수는 반환되는 토큰을 인덱싱하는 부분을 정의합니다.
    소스의 경우 `place=0`, 타겟의 경우 `place=1`입니다
    """
    for english, german in data_iter:
        if place == 0:
            yield engTokenize(english)
        else:
            yield deTokenize(german)

# %%
# 이제 소스 어휘를 빌드하겠습니다

source_vocab = build_vocab_from_iterator(
    getTokens(data_pipe,0),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
source_vocab.set_default_index(source_vocab['<unk>'])

# %%
# 위 코드는 반복자에서 어휘를 만듭니다.
#
# * 2번째 줄에서, 소스 문장 어휘가 필요하므로, `getTokens()` 함수를 `place=0`와 함께 호출합니다
# * 3번째 줄에서, `min_freq=2`로 설정합니다. 이 뜻은 단어가 2번 이하로 나오는 경우 건너뜁니다
# * 4번째 줄에서, 몇 가지 특수 토큰을 지정합니다
#
#   * `<sos>` 문장의 시작
#   * `<eos>` 문장의 끝
#   * `<unk>` 알수없는 단어. 알 수 없는 단어의 예는 `min_freq=2` 같은 이유로 건너뛴 단어입니다
#   * `<pad>` 패딩 토큰. 훈련중에는 모델은 주로 배치로 학습됩니다.
#     배치에서는 길이가 다른 문장이 있을 수 있습니다. 따라서 더 짧은 문장에 패딩을 추가하기 위해,
#     `<pad>` 토큰을 추가하여 배치에 포함된 모든 시퀀스의 길이를 동일하게 만듭니다.
#
# * 5번째 줄에서, `special_first=True`로 설정합니다. 이 의미는 `<pad>`는 인덱스 0, `<sos>`는 인덱스 1을 얻습니다.
#   `<eos>`는 인덱스 2, <unk>는 인덱스 3을 어휘에서 얻게 됩니다.
# * 7번째 줄에서, 기본 인덱스를 `<unk>` 인덱스로 설정했습니다. 즉, 어떤 단어가
#   어휘에 없다면, 그 알 수 없는 단어 대신 `<unk>`를 사용합니다.
#
# 비슷하게, 타겟 문장에 대한 어휘를 구축합니다

target_vocab = build_vocab_from_iterator(
    getTokens(data_pipe,1),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
target_vocab.set_default_index(target_vocab['<unk>'])

# %%
# 어휘에 특수 토큰을 추가하는 방법에 관한 위의 예제를 눈여겨보세요.
# 특수 토큰은 요구 사항에 따라 변경될 수 있습니다.
#
# 이제 특수 토큰이 처음과 다른 단어에 놓이는 것을 확인할 수 있습니다.
# 아래 코드에서, `source_vocab.get_itos()`는 어휘 기반으로 인덱싱된 토큰의 목록을 반환합니다.

print(source_vocab.get_itos()[:9])

# %%
# 어휘를 사용하여 문장을 수치화하기
# ---------------------------------------
# 어휘를 구축한 후에는,  문장을 해당 인덱스로 변환해야 합니다
# 이를 위한 몇 가지 함수를 정의해 보겠습니다

def getTransform(vocab):
    """
    주어진 어휘를 기반으로 transform을 생성합니다. 리턴되는 transform은 토큰 시퀸스에 적용됩니다
    """
    text_tranform = T.Sequential(
        ## 주어진 어휘를 기반으로 문장을 인덱스로 변환합니다
        T.VocabTransform(vocab=vocab),
        ##  각 문장의 시작 부분에 <sos>를 추가합니다. 1 은 이전 섹션에서 보였듯이 <sos>의 인덱스가
        # 1이기 때문입니다
        T.AddToken(1, begin=True),
        ##  각 문장의 시작 부분에 <eos>를 추가합니다. 2 는 이전 섹션에서 보였듯이 <eos>의 인덱스가
        # 2이기 때문입니다
        T.AddToken(2, begin=False)
    )
    return text_tranform

# %%
# 이제 위의 함수를 사용하는 방법을 살펴보겠습니다.
# 이 함수는 문장에 사용할 `Transforms`의 객체를 반환합니다
# 임의의 문장을 가져와서 transform이 어떻게 작동하는지 확인해 보겠습니다.

temp_list = list(data_pipe)
some_sentence = temp_list[798][0]
print("Some sentence=", end="")
print(some_sentence)
transformed_sentence = getTransform(source_vocab)(engTokenize(some_sentence))
print("Transformed sentence=", end="")
print(transformed_sentence)
index_to_string = source_vocab.get_itos()
for index in transformed_sentence:
    print(index_to_string[index], end=" ")

# %%
# 위 코드에서,
#
# * 2번째 줄에서, 1번째 줄의 `data_pipe`에서 생성한 리스트로 된 소스 문장을 가져옵니다.
# * 5번째 줄에서, 소스 어휘를 기반한 transform을 가져와 토큰화된 문장에 적용합니다.
#   변환은 문장이 아닌 단어 리스트를 가져간다는 점에 유의하세요
# * 5번째 줄에서, 인덱싱된 문자열의 맵핑을 가져온 다음 이를 사용하여 transform 적용된 문장을 얻습니다
#
# 이제 DataPipe 함수를 사용하여 모든 문장에 transform을 적용하겠습니다.
# 이를 위해 몇 가지 함수를 더 정의해 보겠습니다.

def applyTransform(sequence_pair):
    """
    시퀀스 쌍의 토큰 시퀀스에 transform을 적용합니다.
    """

    return (
        getTransform(source_vocab)(engTokenize(sequence_pair[0])),
        getTransform(target_vocab)(deTokenize(sequence_pair[1]))
    )
data_pipe = data_pipe.map(applyTransform) ## 반복자의 각 요소에 함수를 적용합니다.
temp_list = list(data_pipe)
print(temp_list[0])

# %%
# 배치 만들기 (버킷 배치와 함께하는)
# -------------------------------------
# 일반적으로 모델을 배치로 훈련합니다. 시퀀스 대 시퀀스 모델에 대해 작업할때는,
# 배치에 포함된 시퀀스의 길이를 비슷하게 유지하는 것이 추천됩니다.
# 이를 위해서 `data_pipe`의 `bucketbatch` 함수를 사용합니다.
#
# `bucketbatch` 함수에서 사용할 몇 가지 함수를 정의해 보겠습니다.

def sortBucket(bucket):
    """
    주어진 버킷을 정렬하는 함수입니다. 여기서는 소스 및 타겟 시퀀스의 길이를
    기준으로 정렬하려고 합니다.
    """
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))

# %%
# 이제 `bucketbatch` 함수를 적용해 보겠습니다

data_pipe = data_pipe.bucketbatch(
    batch_size = 4, batch_num=5,  bucket_num=1,
    use_in_batch_shuffle=False, sort_key=sortBucket
)

# %%
# 위의 코드 블록에서
#
#   * 배치 크기를 4로 유지합니다.
#   * `batch_num`은 버킷에 보관할 배치의 수입니다
#   * `bucket_num`은 셔플을 위해 풀에 보관할 버킷의 수입니다.
#   * `sort_key`는 버킷을 가져와서 정렬하는 함수를 지정합니다.
#
# 이제 소스 문장 묶음을 `X`로, 타겟 문장 묶음을 `y`로 가정해 보겠습니다.
# 일반적으로 모델을 학습할 때는 `X`의 배치에 대해 예측하고 그 결과를 `y`와 비교합니다.
# 하지만 이번 `data_pipe`의 배치는 `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`의 형식입니다.

print(list(data_pipe)[0])
# %%
# 이제 아래와 같은 형식으로 변환해 보겠습니다: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`
# 이를 위해 작은 함수를 작성하겠습니다

def separateSourceTarget(sequence_pairs):
    """
    입력 형식: `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`
    출력 형식: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`
    """
    sources,targets = zip(*sequence_pairs)
    return sources,targets

## 반복자의 각 요소에 함수를 적용합니다
data_pipe = data_pipe.map(separateSourceTarget)
print(list(data_pipe)[0])

# %%
# 이제 원하는 데이터를 얻었습니다
#
# 패딩
# -------
# 앞서 설명한 것처럼 어휘를 구축할 때, 배치에서 모든 시퀀스를 동일한 길이로 만들기 위해
# 짧은 문장은 패딩하게 됩니다. 패딩은 다음과 같이 수행할 수 있습니다.

def applyPadding(pair_of_sequences):
    """
    시퀀스를 tensor로 변환하고 패딩을 적용합니다
    """
    return (T.ToTensor(0)(list(pair_of_sequences[0])), T.ToTensor(0)(list(pair_of_sequences[1])))
## `T.ToTensor(0)`는 시퀀스를 `torch.tensor`로 변환하는 transform을 반환하고 또한 패딩도 적용합니다.
# 여기서 `0`은 생성자에 전달되어, 어휘에 있는`<pad>` 토큰의 인덱스를 지정합니다
data_pipe = data_pipe.map(applyPadding)

# %%
# 이제 인덱스 대신에,
# 인덱싱된 문자열 매핑을 사용하여 토큰화된 시퀀스가 어떻게 보이는지 확인할 수 있습니다.

source_index_to_string = source_vocab.get_itos()
target_index_to_string = target_vocab.get_itos()

def showSomeTransformedSentences(data_pipe):
    """
    모든 transform을 적용한 후 문장이 어떻게 보이는지 보여주는 함수입니다.
    여기서는 해당 인덱스 대신 실제 단어를 출력하려고 합니다.
    """
    for sources,targets in data_pipe:
        if sources[0][-1] != 0:
            continue # 짧은 문장의 패딩만 보이기 위해
        for i in range(4):
            source = ""
            for token in sources[i]:
                source += " " + source_index_to_string[token]
            target = ""
            for token in targets[i]:
                target += " " + target_index_to_string[token]
            print(f"Source: {source}")
            print(f"Traget: {target}")
        break

showSomeTransformedSentences(data_pipe)
# %%
# 위의 출력에서 짧은 문장이 `<pad>`로 채워진 것을 관찰 할 수 있습니다.
# 이제, 훈련 함수를 작성하는 동안 `data_pipe`를 사용할 수 있습니다.
#
# 튜토리얼의 일부는 `이 문서
# <https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71>`__
# 에서 영향을 받았습니다.
