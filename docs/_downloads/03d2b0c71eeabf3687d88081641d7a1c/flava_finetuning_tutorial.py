# -*- coding: utf-8 -*-
"""
TorchMultimodal 튜토리얼: FLAVA 미세조정
============================================

**번역:** `김찬 <https://github.com/chanmuzi>`__

"""


######################################################################
# 멀티 모달 AI는 최근에 이미지 자막추가, 시각적 검색부터 텍스트로부터 이미지를 생성같은
# 최근의 응용까지 그 사용이 빠르게 확산되고 있습니다. **TorchMultimodal은 PyTorch를
# 기반으로 하는 라이브러리로, 멀티 모달 연구를 가능하게 하고 가속화하기 위한 빌딩 블록과
# end-to-end 예제들을 제공합니다**.
#
# 본 튜토리얼에서는 **사전 훈련된 SoTA 모델인** `FLAVA <https://arxiv.org/pdf/2112.04482.pdf>`__ **를**
# **TorchMultimodal 라이브러리에서 사용하여 멀티 모달 작업인 시각적 질의 응답(VQA)에 미세조정하는 방법을 보여 드리겠습니다.**
# 이 모델은 텍스트와 이미지를 위한 두 개의 단일 모달 트랜스포머 기반 인코더와
# 두 임베딩을 결합하는 다중 모달 인코더로 구성되어 있습니다.
# 이 모델은 대조적, 이미지-텍스트 매칭, 그리고 텍스트, 이미지 및 다중 모달 마스킹 손실을 사용하여 사전 훈련되었습니다.



######################################################################
# 설치
# -----------------
# 이 튜토리얼을 위해서는 TextVQA 데이터셋과 Hugging Face의 ``bert 토크나이저`` 를 사용할 것입니다.
# 따라서 TorchMultimodal 외에도 datasets과 transformers를 설치해야 합니다.
#
# .. note::
#    
#    이 튜토리얼을 Google Colab에서 실행할 경우, 새로운 셀을 만들고 다음의 명령어를 실행하여
#    필요한 패키지를 설치하세요:
#
#    .. code-block::
#
#       !pip install torchmultimodal-nightly
#       !pip install datasets
#       !pip install transformers
#

######################################################################
# 단계
# -----
#
# 1. 다음 명령어를 실행하여 Hugging Face 데이터셋을 컴퓨터의 디렉토리에 다운로드하세요:
#
#    .. code-block::
#
#       wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz 
#       tar xf vocab.tar.gz
#
#    .. note:: 
#       이 튜토리얼을 Google Colab에서 실행하는 경우, 새 셀에서 이 명령어를 실행하고 명령어 앞에 느낌표 (!)를 붙이세요.
#
#
# 2. 본 튜토리얼에서는 VQA를 이미지와 질문(텍스트)이 입력되고 출력이 답변 클래스인 분류 작업으로 취급합니다.
#    따라서 답변 클래스와 레이블 매핑을 생성할 단어장 파일을 다운로드해야 합니다.
#
#    또한 Hugging Face에서 `textvqa 데이터셋 <https://arxiv.org/pdf/1904.08920.pdf>`__ 을 불러오는데, 
#    이 데이터셋은 34602개의 훈련 샘플(이미지, 질문, 답변)을 포함하고 있습니다.
#
# 3997개의 답변 클래스가 있음을 확인할 수 있으며, 이에는 알 수 없는 답변을 나타내는 클래스도 포함되어 있습니다.
#

with open("data/vocabs/answers_textvqa_more_than_1.txt") as f:
  vocab = f.readlines()

answer_to_idx = {}
for idx, entry in enumerate(vocab):
  answer_to_idx[entry.strip("\n")] = idx
print(len(vocab))
print(vocab[:5])

from datasets import load_dataset
dataset = load_dataset("facebook/textvqa")

######################################################################
# 데이터셋에서 샘플 엔트리를 표시해 봅시다:
#

import matplotlib.pyplot as plt
import numpy as np 
idx = 5 
print("Question: ", dataset["train"][idx]["question"]) 
print("Answers: " ,dataset["train"][idx]["answers"])
im = np.asarray(dataset["train"][idx]["image"].resize((500,500)))
plt.imshow(im)
plt.show()


######################################################################
# 3. 다음으로, 이미지와 텍스트를 모델에서 사용할 수 있는 텐서로 변환하기 위한 변환 함수를 작성합니다. 
# - 이미지의 경우, torchvision의 변환을 사용하여 텐서로 변환하고 일정한 크기로 조정합니다. 
# - 텍스트의 경우, Hugging Face의 ``BertTokenizer`` 를 사용하여 토큰화(및 패딩)합니다. 
# - 답변(즉, 레이블)의 경우, 가장 빈번하게 나타나는 답변을 훈련 레이블로 사용합니다:
#

import torch
from torchvision import transforms
from collections import defaultdict
from transformers import BertTokenizer
from functools import partial

def transform(tokenizer, input):
  batch = {}
  image_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224])])
  image = image_transform(input["image"][0].convert("RGB"))
  batch["image"] = [image]

  tokenized=tokenizer(input["question"],return_tensors='pt',padding="max_length",max_length=512)
  batch.update(tokenized)


  ans_to_count = defaultdict(int)
  for ans in input["answers"][0]:
    ans_to_count[ans] += 1
  max_value = max(ans_to_count, key=ans_to_count.get)
  ans_idx = answer_to_idx.get(max_value,0)
  batch["answers"] = torch.as_tensor([ans_idx])
  return batch

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased",padding="max_length",max_length=512)
transform=partial(transform,tokenizer)
dataset.set_transform(transform)


######################################################################
# 4. 마지막으로, ``torchmultimodal`` 에서 ``flava_model_for_classification`` 을 가져옵니다.
# 이것은 기본적으로 사전 훈련된 FLAVA 체크포인트를 로드하고 분류 헤드를 포함합니다.
#
# 모델의 순방향 함수는 이미지를 시각 인코더에 통과시키고 질문을 텍스트 인코더에 통과시킵니다.
# 이미지와 질문의 임베딩은 그 후 멀티 모달 인코더를 통과합니다.
# 최종 임베딩은 CLS 토큰에 해당하며, 이는 MLP 헤드를 통과하여 각 가능한 답변에 대한 확률 분포를 제공합니다.
#

from torchmultimodal.models.flava.model import flava_model_for_classification
model = flava_model_for_classification(num_classes=len(vocab))


######################################################################
# 5. 데이터셋과 모델을 함께 모아 3회 반복을 위한 간단한 훈련 루프를 작성하여 
# 모델 훈련 방법을 보여줍니다:
#

from torch import nn
BATCH_SIZE = 2
MAX_STEPS = 3
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset["train"], batch_size= BATCH_SIZE)
optimizer = torch.optim.AdamW(model.parameters())


epochs = 1
for _ in range(epochs):
  for idx, batch in enumerate(train_dataloader):
    optimizer.zero_grad()
    out = model(text = batch["input_ids"], image = batch["image"], labels = batch["answers"])
    loss = out.loss
    loss.backward()
    optimizer.step()
    print(f"Loss at step {idx} = {loss}")
    if idx >= MAX_STEPS-1:
      break


######################################################################
# 결론
# -------------------
#
# 이 튜토리얼에서는 TorchMultimodal의 FLAVA를 사용하여 멀티 모달 작업에 미세 조정하는
# 기본적인 방식을 소개했습니다. 객체 탐지를 위한 멀티 모달 모델인 `MDETR <https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/models/mdetr>`__ 과
# 이미지, 비디오, 3D 분류를 포괄하는 다작업 모델 `Omnivore <https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/omnivore.py>`__
# 같은 라이브러리의 다른 예제들도 확인해 보세요.
# 
#
