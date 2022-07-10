"""
`파이토치(PyTorch) 기본 익히기 <intro.html>`_ ||
`빠른 시작 <quickstart_tutorial.html>`_ ||
**텐서(Tensor)** ||
`Dataset과 Dataloader <data_tutorial.html>`_ ||
`변형(Transform) <transforms_tutorial.html>`_ ||
`신경망 모델 구성하기 <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`최적화(Optimization) <optimization_tutorial.html>`_ ||
`모델 저장하고 불러오기 <saveloadrun_tutorial.html>`_

텐서(Tensor)
==========================================================================

텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료구조입니다.
PyTorch에서는 텐서를 사용하여 모델의 입력(input)과 출력(output), 그리고 모델의 매개변수들을 부호화(encode)합니다.

텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있다는 점만 제외하면 `NumPy <https://numpy.org>`_ 의 ndarray와 유사합니다.
실제로 텐서와 NumPy 배열(array)은 종종 동일한 내부(underly) 메모리를 공유할 수 있어 데이터를 복수할 필요가 없습니다. (:ref:`bridge-to-np-label` 참고)
텐서는 또한 (`Autograd <autogradqs_tutorial.html>`__ 장에서 살펴볼) 자동 미분(automatic differentiation)에 최적화되어 있습니다.
ndarray에 익숙하다면 Tensor API를 바로 사용할 수 있을 것입니다. 아니라면, 아래 내용을 함께 보시죠!
"""

import torch
import numpy as np


######################################################################
# 텐서(tensor) 초기화
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 텐서는 여러가지 방법으로 초기화할 수 있습니다. 다음 예를 살펴보세요:
#
# **데이터로부터 직접(directly) 생성하기**
#
# 데이터로부터 직접 텐서를 생성할 수 있습니다. 데이터의 자료형(data type)은 자동으로 유추합니다.

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

######################################################################
# **NumPy 배열로부터 생성하기**
#
# 텐서는 NumPy 배열로 생성할 수 있습니다. (그 반대도 가능합니다 - :ref:`bridge-to-np-label` 참고)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


###############################################################
# **다른 텐서로부터 생성하기:**
#
# 명시적으로 재정의(override)하지 않는다면, 인자로 주어진 텐서의 속성(모양(shape), 자료형(datatype))을 유지합니다.

x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")


######################################################################
# **무작위(random) 또는 상수(constant) 값을 사용하기:**
#
# ``shape`` 은 텐서의 차원(dimension)을 나타내는 튜플(tuple)로, 아래 함수들에서는 출력 텐서의 차원을 결정합니다.

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



######################################################################
# --------------
#

######################################################################
# 텐서의 속성(Attribute)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 텐서의 속성은 텐서의 모양(shape), 자료형(datatype) 및 어느 장치에 저장되는지를 나타냅니다.

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


######################################################################
# --------------
#

######################################################################
# 텐서 연산(Operation)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 전치(transposing), 인덱싱(indexing), 슬라이싱(slicing), 수학 계산, 선형 대수,
# 임의 샘플링(random sampling) 등, 100가지 이상의 텐서 연산들을
# `여기 <https://pytorch.org/docs/stable/torch.html>`__ 에서 확인할 수 있습니다.
#
# 각 연산들은 (일반적으로 CPU보다 빠른) GPU에서 실행할 수 있습니다. Colab을 사용한다면,
# Edit > Notebook Settings 에서 GPU를 할당할 수 있습니다.
#
# 기본적으로 텐서는 CPU에 생성됩니다. ``.to`` 메소드를 사용하면 (GPU의 가용성(availability)을 확인한 뒤)
# GPU로 텐서를 명시적으로 이동할 수 있습니다. 장치들 간에 큰 텐서들을 복사하는 것은 시간과 메모리 측면에서 비용이
# 많이든다는 것을 기억하세요!

# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


######################################################################
# 목록에서 몇몇 연산들을 시도해보세요.
# NumPy API에 익숙하다면 Tensor API를 사용하는 것은 식은 죽 먹기라는 것을 알게 되실 겁니다.
#

###############################################################
# **NumPy식의 표준 인덱싱과 슬라이싱:**

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

######################################################################
# **텐서 합치기** ``torch.cat`` 을 사용하여 주어진 차원에 따라 일련의 텐서를 연결할 수 있습니다.
# ``torch.cat`` 과 미묘하게 다른 또 다른 텐서 결합 연산인
# `torch.stack <https://pytorch.org/docs/stable/generated/torch.stack.html>`__ 도 참고해보세요.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


######################################################################
# **산술 연산(Arithmetic operations)**

# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


######################################################################
# **단일-요소(single-element) 텐서** 텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 텐서의 경우,
# ``item()`` 을 사용하여 Python 숫자 값으로 변환할 수 있습니다:

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


######################################################################
# **바꿔치기(in-place) 연산**
# 연산 결과를 피연산자(operand)에 저장하는 연산을 바꿔치기 연산이라고 부르며, ``_`` 접미사를 갖습니다.
# 예를 들어: ``x.copy_(y)`` 나 ``x.t_()`` 는 ``x`` 를 변경합니다.

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

######################################################################
# .. note::
#      바꿔치기 연산은 메모리를 일부 절약하지만, 기록(history)이 즉시 삭제되어 도함수(derivative) 계산에 문제가 발생할 수 있습니다.
#      따라서, 사용을 권장하지 않습니다.



######################################################################
# --------------
#


######################################################################
# .. _bridge-to-np-label:
#
# NumPy 변환(Bridge)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPU 상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경됩니다.


######################################################################
# 텐서를 NumPy 배열로 변환하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

######################################################################
# 텐서의 변경 사항이 NumPy 배열에 반영됩니다.

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


######################################################################
# NumPy 배열을 텐서로 변환하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n = np.ones(5)
t = torch.from_numpy(n)

######################################################################
# NumPy 배열의 변경 사항이 텐서에 반영됩니다.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
