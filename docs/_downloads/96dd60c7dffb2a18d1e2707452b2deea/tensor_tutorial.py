"""
Tensor
=======

PyTorch에서의 Tensor는 Torch에서와 거의 동일하게 동작합니다.

초기화되지 않은 (5 x 7) 크기의 tensor를 생성합니다:

"""

import torch
a = torch.empty(5, 7, dtype=torch.float)

###############################################################
# 평균 0, 분산 1의 정규분포를 따르는 무작위 숫자로 dobule tensor를 초기화합니다:

a = torch.randn(5, 7, dtype=torch.double)
print(a)
print(a.size())

###############################################################
# .. note::
#     ``torch.Size`` 는 튜플(tuple)과 같으며, 모든 튜플 연산에 사용할 수 있습니다.
#
# In-place / Out-of-place
# ------------------------
#
# 첫 번째 차이점은 tensor의 모든 In-place 연산은 ``_`` 접미사를 갖는다는 것입니다.
# 예를 들어, ``add`` 는 연산 결과를 돌려주는 Out-of-place 연산을 하고, ``add_`` 는
# In-place 연산을 합니다.

a.fill_(3.5)
# a가 값 3.5로 채워집니다.

b = a.add(4.0)
# a는 여전히 3.5입니다.
# 3.5 + 4.0 = 7.5의 값이 반환되어 새로운 tensor b가 됩니다.

print(a, b)

###############################################################
# ``narrow`` 와 같은 일부 연산들은 In-place 형태를 갖지 않기 때문에 ``.narrow_`` 는
# 존재하지 않습니다. 또한, ``fill_`` 은 Out-of-place 형태를 갖지 않기 떄문에 역시
# ``.fill`` 도 존재하지 않습니다.
#
# 0-인덱스(Zero Indexing)
# -----------------------
#
# 또 다른 차이점은 Tensor의 인덱스는 0부터 시작(0-인덱스)는 점입니다.
# (Lua에서 tensor는 1-인덱스를 갖습니다.)

b = a[0, 3]  # select 1st row, 4th column from a

###############################################################
# Python의 슬라이싱(slicing)으로도 Tensor를 인덱스 할 수 있습니다.

b = a[:, 3:5]  # selects all rows, 4th column and  5th column from a

###############################################################
# 카멜표기법(Camel Case) 없음
# ----------------------------
#
# 그 외에도 카멜표기법을 사용하지 않는 사소한 차이가 있습니다.
# 예를 들어 ``indexAdd`` 는 ``index_add_`` 라고 표기합니다.


x = torch.ones(5, 5)
print(x)

###############################################################
#

z = torch.empty(5, 2)
z[:, 0] = 10
z[:, 1] = 100
print(z)

###############################################################
#
x.index_add_(1, torch.tensor([4, 0], dtype=torch.long), z)
print(x)

###############################################################
# NumPy 변환(Bridge)
# ------------------
#
# Torch Tensor를 NumPy 배열(array)로 변환하거나, 그 반대로 하는 것은 매우 쉽습니다.
# Torch Tensor와 NumPy 배열은 저장 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도
# 변경됩니다.
#
# Torch Tensor를 NumPy 배열로 변환하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

a = torch.ones(5)
print(a)

###############################################################
#

b = a.numpy()
print(b)

###############################################################
#
a.add_(1)
print(a)
print(b) 	# see how the numpy array changed in value


###############################################################
# NumPy 배열을 Torch Tensor로 변환하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)  # see how changing the np array changed the torch Tensor automatically

###############################################################
# CharTensor를 제외한 CPU 상의 모든 Tensor는 NumPy로의 변환을 지원하며,
# (NumPy에서 Tensor로의) 반대 변환도 지원합니다.
#
# CUDA Tensors
# ------------
#
# PyTorch에서 CUDA Tensor는 멋지고 쉽습니다. 그리고 CUDA Tensor를 CPU에서 GPU로 옮겨도
# 기본 형식(underlying type)은 유지됩니다.

# 이 코드는 CUDA가 사용 가능한 환경에서만 실행됩니다.
if torch.cuda.is_available():

    # LongTensor를 생성하고 이를 torch.cuda.LongTensor로 GPU로 옮깁니다.
    a = torch.full((10,), 3, device=torch.device("cuda"))
    print(type(a))
    b = a.to(torch.device("cpu"))
    # CPU로 다시 전송을 하면, torch.LongTensor로 되돌아옵니다.
