# -*- coding: utf-8 -*-
"""
PyTorch가 무엇인가요?
=====================

Python 기반의 과학 연산 패키지로 다음과 같은 두 집단을 대상으로 합니다:

- NumPy를 대체하면서 GPU를 이용한 연산이 필요한 경우
- 최대한의 유연성과 속도를 제공하는 딥러닝 연구 플랫폼이 필요한 경우

시작하기
--------

Tensors
^^^^^^^

Tensor는 NumPy의 ndarray와 유사할 뿐만 아니라, GPU를 사용한 연산 가속도 지원합니다.
"""

from __future__ import print_function
import torch

###############################################################
# 초기화되지 않은 5x3 행렬을 생성합니다:

x = torch.empty(5, 3)
print(x)

###############################################################
# 무작위로 초기화된 행렬을 생성합니다:

x = torch.rand(5, 3)
print(x)

###############################################################
# dtype이 long이고 0으로 채워진 행렬을 생성합니다:

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

###############################################################
# 데이터로부터 바로 tensor를 생성합니다:

x = torch.tensor([5.5, 3])
print(x)

###############################################################
# 또는, 존재하는 tensor를 바탕으로 tensor를 만듭니다. 이 메소드(method)들은
# dtype과 같이 사용자로부터 제공된 새로운 값이 없는 한 입력 tensor의 속성을
# 재사용합니다.

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

###############################################################
# 행렬의 크기를 구합니다:

print(x.size())

###############################################################
# .. note::
#     ``torch.Size`` 는 튜플(tuple)과 같으며, 모든 튜플 연산에 사용할 수 있습니다.
#
# 연산(Operations)
# ^^^^^^^^^^^^^^^^
# 연산을 위한 여러가지 문법을 제공합니다. 다음 예제들을 통해 덧셈 연산을 살펴보겠습니다.
#
# 덧셈: 문법1
y = torch.rand(5, 3)
print(x + y)

###############################################################
# 덧셈: 문법2

print(torch.add(x, y))

###############################################################
# 덧셈: 결과 tensor를 인자로 제공
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

###############################################################
# 덧셈: 바꿔치기(In-place) 방식

# y에 x 더하기
y.add_(x)
print(y)

###############################################################
# .. note::
#     바꿔치기(In-place) 방식으로 tensor의 값을 변경하는 연산은 ``_`` 를 접미사로
#     갖습니다.
#     예: ``x.copy_(y)``, ``x.t_()`` 는 ``x`` 를 변경합니다.
#
# NumPy의 인덱싱 표기 방법을 사용할 수도 있습니다!

print(x[:, 1])

###############################################################
# 크기 변경: tensor의 크기(size)나 모양(shape)을 변경하고 싶을 때, ``torch.view`` 를 사용합니다.
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # 사이즈가 -1인 경우 다른 차원들을 사용하여 유추합니다.
print(x.size(), y.size(), z.size())

###############################################################
# 만약 tensor에 하나의 값만 존재한다면, ``.item()`` 을 사용하면 숫자 값을 얻을 수 있습니다.
x = torch.randn(1)
print(x)
print(x.item())

###############################################################
# **더 읽을거리:**
#
#
#   전치(transposing), 인덱싱(indexing), 슬라이싱(slicing), 수학 계산,
#   선형 대수, 난수(random number) 등과 같은 100가지 이상의 Tensor 연산은
#   `여기 <http://pytorch.org/docs/torch>`_ 에 설명되어 있습니다.
#
# NumPy 변환(Bridge)
# ------------------
#
# Torch Tensor를 NumPy 배열(array)로 변환하거나, 그 반대로 하는 것은 매우 쉽습니다.
#
# Torch Tensor와 NumPy 배열은 저장 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도
# 변경됩니다.
#
# Torch Tensor를 NumPy 배열로 변환하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

a = torch.ones(5)
print(a)

###############################################################
#

b = a.numpy()
print(b)

###############################################################
# NumPy 배열의 값이 어떻게 변하는지 확인해보세요.

a.add_(1)
print(a)
print(b)

###############################################################
# NumPy 배열을 Torch Tensor로 변환하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# NumPy(np) 배열을 변경하면 Torch Tensor의 값도 자동 변경되는 것을 확인해보세요.

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

###############################################################
# CharTensor를 제외한 CPU 상의 모든 Tensor는 NumPy로의 변환을 지원하며,
# (NumPy에서 Tensor로의) 반대 변환도 지원합니다.
#
# CUDA Tensors
# ------------
#
# ``.to`` 메소드를 사용하여 Tensor를 어떠한 장치로도 옮길 수 있습니다.

# 이 코드는 CUDA가 사용 가능한 환경에서만 실행합니다.
# ``torch.device`` 를 사용하여 tensor를 GPU 안팎으로 이동해보겠습니다.
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA 장치 객체(Device Object)로
    y = torch.ones_like(x, device=device)  # GPU 상에 바로(directly) tensor를 생성하거나
    x = x.to(device)                       # 단지 ``.to("cuda")`` 라고만 작성하면 됩니다.
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 는 dtype도 함께 변경합니다!
