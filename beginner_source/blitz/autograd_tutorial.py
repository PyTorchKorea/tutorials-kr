# -*- coding: utf-8 -*-
"""
Autograd: 자동 미분
===================================

PyTorch의 모든 신경망의 중심에는 ``autograd`` 패키지가 있습니다.
먼저 이것을 가볍게 살펴본 뒤, 첫번째 신경망을 학습시켜보겠습니다.

``autograd`` 패키지는 Tensor의 모든 연산에 대해 자동 미분을 제공합니다.
이는 실행-기반-정의(define-by-run) 프레임워크로, 이는 코드를 어떻게 작성하여
실행하느냐에 따라 역전파가 정의된다는 뜻이며, 역전파는 학습 과정의 매 단계마다
달라집니다.

더 간단한 용어로 몇 가지 예를 살펴보겠습니다.

Tensor
--------

패키지의 중심에는 ``torch.Tensor`` 클래스가 있습니다. 만약 ``.requires_grad``
속성을 ``True`` 로 설정하면, 그 tensor에서 이뤄진 모든 연산들을 추적(track)하기
시작합니다. 계산이 완료된 후 ``.backward()`` 를 호출하여 모든 변화도(gradient)를
자동으로 계산할 수 있습니다. 이 Tensor의 변화도는 ``.grad`` 속성에 누적됩니다.

Tensor가 기록을 추적하는 것을 중단하게 하려면, ``.detach()`` 를 호출하여 연산
기록으로부터 분리(detach)하여 이후 연산들이 추적되는 것을 방지할 수 있습니다.

기록을 추적하는 것(과 메모리를 사용하는 것)을 방지하기 위해, 코드 블럭을
``with torch.no_grad():`` 로 감쌀 수 있습니다. 이는 특히 변화도(gradient)는
필요없지만, `requires_grad=True` 가 설정되어 학습 가능한 매개변수를 갖는 모델을
평가(evaluate)할 때 유용합니다.

Autograd 구현에서 매우 중요한 클래스가 하나 더 있는데, 이것은 바로 ``Function``
클래스입니다.

``Tensor`` 와 ``Function`` 은 서로 연결되어 있으며, 모든 연산 과정을
부호화(encode)하여 순환하지 않는 그래프(acyclic graph)를 생성합니다. 각 tensor는
``.grad_fn`` 속성을 갖고 있는데, 이는 ``Tensor`` 를 생성한 ``Function`` 을
참조하고 있습니다. (단, 사용자가 만든 Tensor는 예외로, 이 때 ``grad_fn`` 은
``None`` 입니다.)

도함수를 계산하기 위해서는 ``Tensor`` 의 ``.backward()`` 를 호출하면
됩니다. 만약 ``Tensor`` 가 스칼라(scalar)인 경우(예. 하나의 요소 값만 갖는 등)에는
``backward`` 에 인자를 정해줄 필요가 없습니다. 하지만 여러 개의 요소를 갖고 있을
때는 tensor의 모양을 ``gradient`` 의 인자로 지정할 필요가 있습니다.
"""

import torch

###############################################################
# tensor를 생성하고 ``requires_grad=True`` 를 설정하여 연산을 기록합니다.
x = torch.ones(2, 2, requires_grad=True)
print(x)

###############################################################
# tensor에 연산을 수행합니다:
y = x + 2
print(y)

###############################################################
# ``y`` 는 연산의 결과로 생성된 것이므로 ``grad_fn`` 을 갖습니다.
print(y.grad_fn)

###############################################################
# ``y`` 에 다른 연산을 수행합니다.
z = y * y * 3
out = z.mean()

print(z, out)

################################################################
# ``.requires_grad_( ... )`` 는 기존 Tensor의 ``requires_grad`` 값을 바꿔치기
# (in-place)하여 변경합니다. 입력값이 지정되지 않으면 기본값은 ``False`` 입니다.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

###############################################################
# 변화도(Gradient)
# -----------------
# 이제 역전파(backprop)를 해보겠습니다.
# ``out`` 은 하나의 스칼라 값만 갖고 있기 때문에, ``out.backward()`` 는
# ``out.backward(torch.tensor(1.))`` 과 동일합니다.

out.backward()

###############################################################
# 변화도 d(out)/dx를 출력합니다.
#

print(x.grad)

###############################################################
# ``4.5`` 로 이루어진 행렬을 확인할 수 있습니다. ``out`` 을 *Tensor* “:math:`o`”
# 라고 하면, 다음과 같이 구할 수 있습니다.
# :math:`o = \frac{1}{4}\sum_i z_i` 이고,
# :math:`z_i = 3(x_i+2)^2` 이므로 :math:`z_i\bigr\rvert_{x_i=1} = 27` 입니다.
# 따라서,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)` 이므로,
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5` 입니다.

###############################################################
# 수학적으로 벡터 함수 :math:`\vec{y}=f(\vec{x})` 에서 :math:`\vec{x}` 에
# 대한 :math:`\vec{y}` 의 변화도는 야코비안 행렬(Jacobian Matrix)입니다:
#
# .. math::
#   J=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)
#
# 일반적으로, ``torch.autograd`` 는 벡터-야코비안 곱을 계산하는 엔진입니다. 즉,
# 어떤 벡터 :math:`v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots & v_{m}\end{array}\right)^{T}`
# 에 대해 :math:`v^{T}\cdot J` 을 연산합니다. 만약 :math:`v` 가 스칼라 함수
# :math:`l=g\left(\vec{y}\right)` 의 기울기인 경우,
# :math:`v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}`
# 이며, 연쇄법칙(chain rule)에 따라 벡터-야코비안 곱은 :math:`\vec{x}` 에 대한
# :math:`l` 의 기울기가 됩니다:
#
# .. math::
#   J^{T}\cdot v=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)\left(\begin{array}{c}
#    \frac{\partial l}{\partial y_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial y_{m}}
#    \end{array}\right)=\left(\begin{array}{c}
#    \frac{\partial l}{\partial x_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial x_{n}}
#    \end{array}\right)
#
# (여기서 :math:`v^{T}\cdot J` 은 :math:`J^{T}\cdot v` 를 취했을 때의 열 벡터로
# 취급할 수 있는 행 벡터를 갖습니다.)
#
# 벡터-야코비안 곱의 이러한 특성은 스칼라가 아닌 출력을 갖는 모델에 외부 변화도를
# 제공(feed)하는 것을 매우 편리하게 해줍니다.

###############################################################
# 이제 벡터-야코비안 곱의 예제를 살펴보도록 하겠습니다:

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

###############################################################
# 이 경우 ``y`` 는 더 이상 스칼라 값이 아닙니다. ``torch.autograd`` 는
# 전체 야코비안을 직접 계산할수는 없지만, 벡터-야코비안 곱은 간단히
# ``backward`` 에 해당 벡터를 인자로 제공하여 얻을 수 있습니다:
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

###############################################################
# 또한 ``with torch.no_grad():`` 로 코드 블럭을 감싸서 autograd가
# ``.requires_grad=True`` 인 Tensor들의 연산 기록을 추적하는 것을 멈출 수 있습니다.
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

###############################################################
# 또는 ``.detach()`` 를 호출하여 내용물(content)은 같지만 require_grad가 다른
# 새로운 Tensor를 가져옵니다:
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())


###############################################################
# **더 읽을거리:**
#
# ``autograd.Function`` 관련 문서는 https://pytorch.org/docs/stable/autograd.html#function
# 에서 찾아볼 수 있습니다.
