# -*- coding: utf-8 -*-
"""
Autograd
========

Autograd는 자동 미분을 수행하는 torch의 핵심 패키지로, 자동 미분을 위해
테잎(tape) 기반 시스템을 사용합니다.

순전파(forward) 단계에서 autograd 테잎은 수행하는 모든 연산을 기억합니다.
그리고, 역전파(backward) 단계에서 연산들을 재현(replay)합니다.

연산 기록을 추적하는 Tensor
----------------------------

Autograd에서 ``requires_grad=True`` 로 설정된 입력 ``Tensor`` 의 연산은
기록됩니다. 역전파 단계 연산 후에, 이 Tensor에 대한 변화도(grdient)는 ``.grad``
속성에 누적됩니다.

Autograd 구현에서 매우 중요한 클래스가 하나 더 있는데, 이것은 바로 ``Function``
클래스입니다. ``Tensor`` 와 ``Function`` 은 서로 연결되어 있으며, 모든 연산 과정을
부호화(encode)하여 순환하지 않는 그래프(acyclic graph)를 생성합니다. 각 변수는
``.grad_fn`` 속성을 갖고 있는데, 이는 ``Tensor`` 를 생성한 ``Function`` 을
참조하고 있습니다. (단, 사용자가 만든 Tensor는 예외로, 이 때 ``grad_fn`` 은
``None`` 입니다.)

도함수를 계산하기 위해서는 ``Tensor`` 의 ``.backward()`` 를 호출하면 됩니다.
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
#
print(x.data)

###############################################################
#
print(x.grad)

###############################################################
#

print(x.grad_fn)  # x는 직접 생성하였기 때문에 아무런 값도 없습니다.

###############################################################
# x에 연산을 수행합니다:

y = x + 2
print(y)

###############################################################
# y 는 연산의 결과로 생성된 것이므로, grad_fn을 갖습니다.
print(y.grad_fn)

###############################################################
# y에 다른 연산을 수행합니다.

z = y * y * 3
out = z.mean()

print(z, out)

################################################################
# ``.requires_grad_( ... )`` 는 기존 Tensor의 ``requires_grad`` 값을 바꿔치기하여
# 변경합니다. 입력값이 지정되지 않으면 기본값은 ``False`` 입니다.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

###############################################################
# 변화도(Gradient)
# ----------------
#
# 이제 역전파를 한 후 변화도 d(out)/dx를 출력해보겠습니다.

out.backward()
print(x.grad)


###############################################################
# 기본적으로 변화도 연산은 그래프 내의 모든 내부 버퍼를 날려버리므로,
# 그래프의 일부를 2번 역전파하려면 첫번째 역전파 시에 미리
# ``retain_graph = True`` 을 지정해둘 필요가 있습니다.

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
y.backward(torch.ones(2, 2), retain_graph=True)
# retain_graph는 내부 버퍼들이 지워지는 것을 막습니다.
print(x.grad)

###############################################################
#
z = y * y
print(z)

###############################################################
#
# 무작위 경사도로 역전파해보겠습니다

gradient = torch.randn(2, 2)

# 만약 앞에서 retain_graph를 하지 않았다면 여기서 에러가 발생할 것입니다.
y.backward(gradient)

print(x.grad)

###############################################################
# 또한 ``with torch.no_grad():`` 로 코드 블럭을 감싸서 autograd가
# ``.requires_grad=True`` 인 Tensor들의 연산 기록을 추적하는 것을 멈출 수 있습니다:
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)
