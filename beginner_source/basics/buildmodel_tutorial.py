"""
`파이토치(PyTorch) 기본 익히기 <intro.html>`_ ||
`빠른 시작 <quickstart_tutorial.html>`_ ||
`텐서(Tensor) <tensorqs_tutorial.html>`_ ||
`Dataset과 Dataloader <data_tutorial.html>`_ ||
`변형(Transform) <transforms_tutorial.html>`_ ||
**신경망 모델 구성하기** ||
`Autograd <autogradqs_tutorial.html>`_ ||
`최적화(Optimization) <optimization_tutorial.html>`_ ||
`모델 저장하고 불러오기 <saveloadrun_tutorial.html>`_

신경망 모델 구성하기
==========================================================================

신경망은 데이터에 대한 연산을 수행하는 계층(layer)/모듈(module)로 구성되어 있습니다.
`torch.nn <https://pytorch.org/docs/stable/nn.html>`_ 네임스페이스는 신경망을 구성하는데 필요한 모든 구성 요소를 제공합니다.
PyTorch의 모든 모듈은 `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ 의 하위 클래스(subclass)
입니다. 신경망은 다른 모듈(계층; layer)로 구성된 모듈입니다. 이러한 중첩된 구조는 복잡한 아키텍처를 쉽게 구축하고 관리할 수 있습니다.

이어지는 장에서는 FashionMNIST 데이터셋의 이미지들을 분류하는 신경망을 구성해보겠습니다.

"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#############################################
# 학습을 위한 장치 얻기
# ------------------------------------------------------------------------------------------
#
# 가능한 경우 GPU와 같은 하드웨어 가속기에서 모델을 학습하려고 합니다.
# `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_ 를 사용할 수 있는지
# 확인하고 그렇지 않으면 CPU를 계속 사용합니다.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

##############################################
# 클래스 정의하기
# ------------------------------------------------------------------------------------------
#
# 신경망 모델을 ``nn.Module`` 의 하위클래스로 정의하고, ``__init__`` 에서 신경망 계층들을 초기화합니다.
# ``nn.Module`` 을 상속받은 모든 클래스는 ``forward`` 메소드에 입력 데이터에 대한 연산들을 구현합니다.

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

##############################################
# ``NeuralNetwork`` 의 인스턴스(instance)를 생성하고 이를 ``device`` 로 이동한 뒤,
# 구조(structure)를 출력합니다.

model = NeuralNetwork().to(device)
print(model)


##############################################
# 모델을 사용하기 위해 입력 데이터를 전달합니다. 이는 일부
# `백그라운드 연산들 <https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866>`_ 과 함께
# 모델의 ``forward`` 를 실행합니다. ``model.forward()`` 를 직접 호출하지 마세요!
#
# 모델에 입력을 전달하여 호출하면 2차원 텐서를 반환합니다. 2차원 텐서의 dim=0은 각 분류(class)에 대한 원시(raw) 예측값 10개가,
# dim=1에는 각 출력의 개별 값들이 해당합니다.
# 원시 예측값을 ``nn.Softmax`` 모듈의 인스턴스에 통과시켜 예측 확률을 얻습니다.

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


######################################################################
# ------------------------------------------------------------------------------------------
#


##############################################
# 모델 계층(Layer)
# ------------------------------------------------------------------------------------------
#
# FashionMNIST 모델의 계층들을 살펴보겠습니다. 이를 설명하기 위해, 28x28 크기의 이미지 3개로 구성된
# 미니배치를 가져와, 신경망을 통과할 때 어떤 일이 발생하는지 알아보겠습니다.

input_image = torch.rand(3,28,28)
print(input_image.size())

##################################################
# nn.Flatten
# ^^^^^^^^^^^^^^^^^^^^^^
# `nn.Flatten  <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_ 계층을 초기화하여
# 각 28x28의 2D 이미지를 784 픽셀 값을 갖는 연속된 배열로 변환합니다. (dim=0의 미니배치 차원은 유지됩니다.)

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

##############################################
# nn.Linear
# ^^^^^^^^^^^^^^^^^^^^^^
# `선형 계층 <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ 은 저장된 가중치(weight)와
# 편향(bias)을 사용하여 입력에 선형 변환(linear transformation)을 적용하는 모듈입니다.
#
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


#################################################
# nn.ReLU
# ^^^^^^^^^^^^^^^^^^^^^^
# 비선형 활성화(activation)는 모델의 입력과 출력 사이에 복잡한 관계(mapping)를 만듭니다.
# 비선형 활성화는 선형 변환 후에 적용되어 *비선형성(nonlinearity)* 을 도입하고, 신경망이 다양한 현상을 학습할 수 있도록 돕습니다.
#
# 이 모델에서는 `nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_ 를 선형 계층들 사이에 사용하지만,
# 모델을 만들 때는 비선형성을 가진 다른 활성화를 도입할 수도 있습니다.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")



#################################################
# nn.Sequential
# ^^^^^^^^^^^^^^^^^^^^^^
# `nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ 은 순서를 갖는
# 모듈의 컨테이너입니다. 데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달됩니다. 순차 컨테이너(sequential container)를 사용하여
# 아래의 ``seq_modules`` 와 같은 신경망을 빠르게 만들 수 있습니다.

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

################################################################
# nn.Softmax
# ^^^^^^^^^^^^^^^^^^^^^^
# 신경망의 마지막 선형 계층은 `nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ 모듈에 전달될
# ([-\infty, \infty] 범위의 원시 값(raw value)인) `logits` 를 반환합니다. logits는 모델의 각 분류(class)에 대한 예측 확률을 나타내도록
# [0, 1] 범위로 비례하여 조정(scale)됩니다. ``dim`` 매개변수는 값의 합이 1이 되는 차원을 나타냅니다.

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


#################################################
# 모델 매개변수
# ------------------------------------------------------------------------------------------
#
# 신경망 내부의 많은 계층들은 *매개변수화(parameterize)* 됩니다. 즉, 학습 중에 최적화되는 가중치와 편향과 연관지어집니다.
# ``nn.Module`` 을 상속하면 모델 객체 내부의 모든 필드들이 자동으로 추적(track)되며, 모델의 ``parameters()`` 및
# ``named_parameters()`` 메소드로 모든 매개변수에 접근할 수 있게 됩니다.
#
# 이 예제에서는 각 매개변수들을 순회하며(iterate), 매개변수의 크기와 값을 출력합니다.
#


print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

######################################################################
# ------------------------------------------------------------------------------------------
#

#################################################################
# 더 읽어보기
# ------------------------------------------------------------------------------------------
# - `torch.nn API <https://pytorch.org/docs/stable/nn.html>`_
