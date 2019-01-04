# -*- coding: utf-8 -*-
"""
Autograd: 자동 미분
===================================

PyTorch의 모든 신경망의 중심에는 ``autograd`` 패키지가 있습니다.
먼저 이것을 가볍게 살펴본 뒤, 첫번째 신경망을 학습시켜보겠습니다.


``autograd`` 패키지는 Tensor의 모든 연산에 대해 자동 미분을 제공합니다.
이는 실행-기반-정의(define-by-run) 프레임워크로, 이는 코드를 어떻게 작성하여 실행하느냐에
따라 역전파가 정의된다는 뜻이며, 역전파는 학습 과정의 매 단계마다 달라집니다.

좀 더 간단한 용어로 몇 가지 예를 살펴보겠습니다.

Tensor
--------

패키지의 중심에는 ``torch.Tensor`` 클래스가 있습니다. 만약 ``.requires_grad``
속성을 ``True`` 로 설정하면, 그 tensor에서 이뤄진 모든 연산들을 추적(Track)하기
시작합니다. 계산이 완료된 후 ``.backward()`` 를 호출하여 모든 변화도(gradient)를
자동으로 계산할 수 있습니다. 이 Tensor의 변화도는 ``.grad`` 에 누적됩니다.

Tensor가 기록을 중단하게 하려면, ``.detach()`` 를 호출하여 연산 기록으로부터
분리(Detach)하여 이후 연산들이 기록되는 것을 방지할 수 있습니다.

연산 기록을 추적하는 것(과 메모리 사용)을 멈추기 위해, 코드 블럭(Code Block)을
``with torch.no_grad():`` 로 감쌀 수 있습니다. 이는 특히 변화도(Gradient)는
필요없지만, `requires_grad=True` 가 설정되어 학습 가능한 매개변수(Parameter)를
갖는 모델을 실행(Evaluate)할 때 유용합니다.

Autograd 구현에서 매우 중요한 클래스가 하나 더 있는데요, 바로 ``Function``
클래스입니다.

``Tensor`` 와 ``Function`` 은 상호 연결되어 있으며, 모든 연산 과정을
부호화(encode)하여 순환하지 않은 그래프(acyclic graph)를 생성합니다. 각 변수는
``.grad_fn`` 속성을 갖고 있는데, 이는 ``Tensor`` 를 생성한 ``Function`` 을
참조하고 있습니다. (단, 사용자가 만든 Tensor는 예외로, 이 때 ``grad_fn`` 은
``None`` 입니다.)

도함수를 계산하기 위해서는, ``Tensor`` 의 ``.backward()`` 를 호출하면 됩니다.
``Tensor`` 가 스칼라(scalar)인 경우(예. 하나의 요소만 갖는 등)에는, ``backward`` 에
인자를 정해줄 필요가 없습니다. 하지만 여러 개의 요소를 갖고 있을
때는 tensor의 모양을 ``gradient`` 의 인자로 지정할 필요가 있습니다.
"""

import torch

###############################################################
# tensor를 생성하고 requires_grad=True를 설정하여 연산을 기록합니다.
x = torch.ones(2, 2, requires_grad=True)
print(x)

###############################################################
# tensor에 연산을 수행합니다:
y = x + 2
print(y)

###############################################################
# ``y`` 는 연산의 결과로 생성된 것이므로, ``grad_fn`` 을 갖습니다.
print(y.grad_fn)

###############################################################
# y에 다른 연산을 수행합니다.
z = y * y * 3
out = z.mean()

print(z, out)

################################################################
# ``.requires_grad_( ... )`` 는 기존 Tensor의 ``requires_grad`` 값을 In-place로
# 변경합니다. 입력값이 지정되지 않으면 기본값은 ``True`` 입니다.
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
# ``out`` 은 하나의 스칼라(Scalar) 값만 갖고 있기 때문에, ``out.backward()`` 는
# ``out.backward(torch.tensor(1))`` 을 하는 것과 똑같습니다.

out.backward()

###############################################################
# 변화도 d(out)/dx를 출력합니다.
#

print(x.grad)

###############################################################
# ``4.5`` 로 이루어진 행렬이 보일 것입니다. ``out`` 을 *Tensor* “:math:`o`” 라고
# 하면, 다음과 같이 구할 수 있습니다.
# :math:`o = \frac{1}{4}\sum_i z_i`,
# :math:`z_i = 3(x_i+2)^2` 이고 :math:`z_i\bigr\rvert_{x_i=1} = 27` 입니다.
# 따라서,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)` 이므로,
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5`.

###############################################################
# autograd로 많은 정신나간 일들(crazy things)도 할 수 있습니다!


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

###############################################################
#
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)

###############################################################
# ``with torch.no_grad():`` 로 코드 블럭(Code Block)을 감싸서, autograd가
# requires_grad=True인 Tensor들의 연산 기록을 추적하는 것을 멈출 수 있습니다.
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

###############################################################
# **더 읽을거리:**
#
# ``Variable`` 과 ``Function`` 관련 문서는 http://pytorch.org/docs/autograd 에
# 있습니다.
