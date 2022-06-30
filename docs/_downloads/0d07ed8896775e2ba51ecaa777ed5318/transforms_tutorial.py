"""
`파이토치(PyTorch) 기본 익히기 <intro.html>`_ ||
`빠른 시작 <quickstart_tutorial.html>`_ ||
`텐서(Tensor) <tensorqs_tutorial.html>`_ ||
`Dataset과 Dataloader <data_tutorial.html>`_ ||
**변형(Transform)** ||
`신경망 모델 구성하기 <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`최적화(Optimization) <optimization_tutorial.html>`_ ||
`모델 저장하고 불러오기 <saveloadrun_tutorial.html>`_

변형(Transform)
==========================================================================

데이터가 항상 머신러닝 알고리즘 학습에 필요한 최종 처리가 된 형태로 제공되지는 않습니다.
**변형(transform)** 을 해서 데이터를 조작하고 학습에 적합하게 만듭니다.

모든 TorchVision 데이터셋들은 변형 로직을 갖는, 호출 가능한 객체(callable)를 받는 매개변수 두개
( 특징(feature)을 변경하기 위한 ``transform`` 과 정답(label)을 변경하기 위한 ``target_transform`` )를 갖습니다
`torchvision.transforms <https://pytorch.org/vision/stable/transforms.html>`_ 모듈은
주로 사용하는 몇가지 변형(transform)을 제공합니다.

FashionMNIST 특징(feature)은 PIL Image 형식이며, 정답(label)은 정수(integer)입니다.
학습을 하려면 정규화(normalize)된 텐서 형태의 특징(feature)과 원-핫(one-hot)으로 부호화(encode)된 텐서 형태의
정답(label)이 필요합니다. 이러한 변형(transformation)을 하기 위해 ``ToTensor`` 와 ``Lambda`` 를 사용합니다.
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

#################################################
# ToTensor()
# ------------------------------------------------------------------------------------------
#
# `ToTensor <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor>`_
# 는 PIL Image나 NumPy ``ndarray`` 를  ``FloatTensor`` 로 변환하고, 이미지의 픽셀의 크기(intensity) 값을 [0., 1.] 범위로
# 비례하여 조정(scale)합니다.
#

##############################################
# Lambda 변형(Transform)
# ------------------------------------------------------------------------------------------
#
# Lambda 변형은 사용자 정의 람다(lambda) 함수를 적용합니다. 여기에서는 정수를 원-핫으로 부호화된 텐서로 바꾸는
# 함수를 정의합니다.
# 이 함수는 먼저 (데이터셋 정답의 개수인) 크기 10짜리 영 텐서(zero tensor)를 만들고,
# `scatter_ <https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html>`_ 를 호출하여
# 주어진 정답 ``y`` 에 해당하는 인덱스에 ``value=1`` 을 할당합니다.

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

######################################################################
# ------------------------------------------------------------------------------------------
#

#################################################################
# 더 읽어보기
# ~~~~~~~~~~~~~~~~~
# - `torchvision.transforms API <https://pytorch.org/vision/stable/transforms.html>`_
