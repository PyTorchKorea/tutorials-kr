# -*- coding: utf-8 -*-
r"""
PyTorch 소개 
***********************

Torch의 텐서(Tensor) 라이브러리 소개
======================================

모든 딥러닝은 2차원 이상으로 색인될 수 있는 행렬의 일반화인 
텐서의 연산입니다. 이것이 무엇을 의미하지 나중에 정확히 
알게 될 것입니다. 먼저, 텐서로 무엇을 할 수 있는지 살펴 봅시다.
"""
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


######################################################################
# 텐서 생성하기
# ~~~~~~~~~~~~~~~~
#
# 텐서는 파이썬 리스트에서 torch.Tensor() 함수로 생성될 수 있습니다.
#

# torch.tensor(data) 는 주어진 데이터로 torch.Tensor 객체를 생성합니다. 
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)

# 행렬 생성
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.tensor(M_data)
print(M)

# 2x2x2 크기의 3D 텐서 생성.
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.tensor(T_data)
print(T)


######################################################################
# 어쨌든 3D 텐서가 무엇입니까? 이렇게 생각해 보십시오. 만약 벡터가 있다면
# 벡터에 주소를 입력하면 스칼라를 줍니다. 만약 행렬을 가지고 있다면 행렬에
# 주소를 입력하면 벡터를 줍니다. 만약 3D 텐서를 가지고 있다면 텐서에 주소를
# 입력하면 행렬을 줍니다.
#
# 용어에 대한 주석:
# 이 튜토리얼에서 "텐서"를 언급 할 때 그것은 어떤 torch.Tensor 객체를 말합니다.
# 행렬과 벡터는 각각 차원이 1과 2인 torch.Tensors 의 특별한 케이스 입니다.
# 3D 텐서를 말할 때는 "3D 텐서"라고 명시적으로 사용하겠습니다.
#

# Index into V and get a scalar (0 dimensional tensor)
print(V[0])
# Get a Python number from it
print(V[0].item())

# Index into M and get a vector
print(M[0])

# Index into T and get a matrix
print(T[0])


######################################################################
# 다른 데이터 유형의 텐서를 생성 할 수도 있습니다. 보시다시피 기본값은 
# Float입니다. 정수형의 텐서를 만들려면 torch.LongTensor ()를 사용하십시오.
# 더 많은 데이터 유형에 대해서는 설명서를 확인하십시오.
# 그러나 Float 및 Long이 가장 일반적입니다.
#


######################################################################
# torch.randn ()을 사용하여 랜덤 데이터와 제공된 차원으로 텐서를 
# 만들 수 있습니다.
#

x = torch.randn((3, 4, 5))
print(x)


######################################################################
# 텐서로 작업하기
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# 기대하는 방식으로 텐서로 작업할 수 있습니다.

x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(z)


######################################################################
# 사용 가능한 방대한 작업의 전체 목록은 
# `문서 <http://pytorch.org/docs/torch.html>`__ 를 참고하십시오. 단순한
# 수학적 연산 이상으로 확장됩니다.
# 
# 나중에 사용하게 될 유용한 작업 중 하나는 연결입니다.
#

# 기본으로 첫번째 축(가로 연결)을 따라 연결합니다.
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# 세로 연결:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# 두번째 변수는 연결될 축을 결정합니다.
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# 텐서가 호환되지 않으면 Torch가 오류 메시지를 출력 합니다. 주석 처리를 제거하여 오류를 확인하십시오.
# torch.cat([x_1, x_2])


######################################################################
# 텐서 재구성
# ~~~~~~~~~~~~~~~~~
#
# .view() 메서드를 사용해서 텐서를 재구성합니다.
# 이 메서드는 많은 신경망 구성 요소가 입력을 특정 모양으로 예상하기 
# 때문에 많이 사용됩니다. 데이터를 구성 요소로 전달하기 전에 종종 모양을
# 변경해야합니다.
#

x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))  # 가로 2 , 세로 12로 재구성 
# 위와 같이 하나의 차원이 -1이면 그 것의 크기는 유추될 수 있습니다.
print(x.view(2, -1))


######################################################################
# 연산 그래프(Computation Graph)와 자동 미분(Automatic Differentiation)
# ================================================
#
# 연산 그래프의 개념은 효율적인 딥러닝 프로그래밍에 필수적입니다. 
# 역전파 그라디언트를 직접 작성할 필요가 없기 때문입니다.

계산 그래프는 단순히 데이터를 결합하여 출력을 제공하는 방법의 스펙입니다.

그래프는 어떤 연산과 관련된 매개 변수를 완전하게 지정하기 때문에 파생물을 계산하기에 충분한 정보를 포함합니다.

아마 모호하게 들릴지도 모르니, 근본적인 플래그``requires_grad``를 어떻게 사용하고 있는지 보자.
# The concept of a computation graph is essential to efficient deep
# learning programming, because it allows you to not have to write the
# back propagation gradients yourself. A computation graph is simply a
# specification of how your data is combined to give you the output. Since
# the graph totally specifies what parameters were involved with which
# operations, it contains enough information to compute derivatives. This
# probably sounds vague, so let's see what is going on using the
# fundamental flag ``requires_grad``.
#
# First, think from a programmers perspective. What is stored in the
# torch.Tensor objects we were creating above? Obviously the data and the
# shape, and maybe a few other things. But when we added two tensors
# together, we got an output tensor. All this output tensor knows is its
# data and shape. It has no idea that it was the sum of two other tensors
# (it could have been read in from a file, it could be the result of some
# other operation, etc.)
#
# If ``requires_grad=True``, the Tensor object keeps track of how it was
# created. Lets see it in action.
#

# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3], requires_grad=True)

# With requires_grad=True, you can still do all the operations you previously
# could
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
print(z)

# BUT z knows something extra.
print(z.grad_fn)


######################################################################
# So Tensors know what created them. z knows that it wasn't read in from
# a file, it wasn't the result of a multiplication or exponential or
# whatever. And if you keep following z.grad_fn, you will find yourself at
# x and y.
#
# But how does that help us compute a gradient?
#

# Lets sum up all the entries in z
s = z.sum()
print(s)
print(s.grad_fn)


######################################################################
# So now, what is the derivative of this sum with respect to the first
# component of x? In math, we want
#
# .. math::
#
#    \frac{\partial s}{\partial x_0}
#
#
#
# Well, s knows that it was created as a sum of the tensor z. z knows
# that it was the sum x + y. So
#
# .. math::  s = \overbrace{x_0 + y_0}^\text{$z_0$} + \overbrace{x_1 + y_1}^\text{$z_1$} + \overbrace{x_2 + y_2}^\text{$z_2$}
#
# And so s contains enough information to determine that the derivative
# we want is 1!
#
# Of course this glosses over the challenge of how to actually compute
# that derivative. The point here is that s is carrying along enough
# information that it is possible to compute it. In reality, the
# developers of Pytorch program the sum() and + operations to know how to
# compute their gradients, and run the back propagation algorithm. An
# in-depth discussion of that algorithm is beyond the scope of this
# tutorial.
#


######################################################################
# Lets have Pytorch compute the gradient, and see that we were right:
# (note if you run this block multiple times, the gradient will increment.
# That is because Pytorch *accumulates* the gradient into the .grad
# property, since for many models this is very convenient.)
#

# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)


######################################################################
# Understanding what is going on in the block below is crucial for being a
# successful programmer in deep learning.
#

x = torch.randn(2, 2)
y = torch.randn(2, 2)
# By default, user created Tensors have ``requires_grad=False``
print(x.requires_grad, y.requires_grad)
z = x + y
# So you can't backprop through z
print(z.grad_fn)

# ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# flag in-place. The input flag defaults to ``True`` if not given.
x = x.requires_grad_()
y = y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
z = x + y
print(z.grad_fn)
# If any input to an operation has ``requires_grad=True``, so will the output
print(z.requires_grad)

# Now z has the computation history that relates itself to x and y
# Can we just take its values, and **detach** it from its history?
new_z = z.detach()

# ... does new_z have information to backprop to x and y?
# NO!
print(new_z.grad_fn)
# And how could it? ``z.detach()`` returns a tensor that shares the same storage
# as ``z``, but with the computation history forgotten. It doesn't know anything
# about how it was computed.
# In essence, we have broken the Tensor away from its past history

###############################################################
# You can also stops autograd from tracking history on Tensors
# with requires_grad=True by wrapping the code block in
# ``with torch.no_grad():``
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)


