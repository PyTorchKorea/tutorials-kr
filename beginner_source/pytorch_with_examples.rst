예제로 배우는 파이토치(PyTorch)
************************************
**Author**: `Justin Johnson <https://github.com/jcjohnson/pytorch-examples>`_
**번역**: `박정환 <https://github.com/9bow>`_

.. Note::
    이 튜토리얼은 다소 오래된 PyTorch 튜토리얼입니다.
    `기본 다지기 <https://tutorials.pytorch.kr/beginner/basics/intro.html>`_ 에서
    입문자를 위한 최신의 내용을 보실 수 있습니다.

이 튜토리얼에서는 `PyTorch <https://github.com/pytorch/pytorch>`__ 의 핵심적인 개념을
예제를 통해 소개합니다.

본질적으로, PyTorch에는 두가지 주요한 특징이 있습니다:

- NumPy와 유사하지만 GPU 상에서 실행 가능한 n-차원 텐서(Tensor)
- 신경망을 구성하고 학습하는 과정에서의 자동 미분(Automatic differentiation)

이 튜토리얼에서는 3차 다항식(third order polynomial)을 사용하여 :math:`y=\sin(x)` 에 근사(fit)하는 문제를 다뤄보겠습니다.
신경망은 4개의 매개변수를 가지며, 정답과 신경망이 예측한 결과 사이의 유클리드 거리(Euclidean distance)를
최소화하여 임의의 값을 근사할 수 있도록 경사하강법(gradient descent)을 사용하여 학습하겠습니다.

.. Note::
    각각의 예제들은 :ref:`이 문서의 마지막 <examples-download>` 에서 살펴볼 수 있습니다.

.. contents:: Table of Contents
	:local:

텐서(Tensor)
=============

준비 운동: numpy
-------------------------------------------------------------------------------

PyTorch를 소개하기 전에, 먼저 NumPy를 사용하여 신경망을 구성해보겠습니다.

NumPy는 n-차원 배열 객체와 이러한 배열들을 조작하기 위한 다양한 함수들을 제공합니다. NumPy는 과학 분야의
연산을 위한 포괄적인 프레임워크(generic framework)입니다;
NumPy는 연산 그래프(computation graph)나 딥러닝, 변화도(gradient)에 대해서는 알지 못합니다.
하지만 NumPy 연산을 사용하여 신경망의 순전파 단계와 역전파 단계를 직접 구현함으로써,
3차 다항식이 사인(sine) 함수에 근사하도록 만들 수 있습니다:

.. includenodoc:: /beginner/examples_tensor/polynomial_numpy.py


파이토치(PyTorch): 텐서(Tensor)
-------------------------------------------------------------------------------

NumPy는 훌륭한 프레임워크지만, GPU를 사용하여 수치 연산을 가속화할 수는 없습니다.
현대의 심층 신경망에서 GPU는 종종 `50배 또는 그 이상 <https://github.com/jcjohnson/cnn-benchmarks>`__ 의
속도 향상을 제공하기 때문에, 안타깝게도 NumPy는 현대의 딥러닝에는 충분치 않습니다.

이번에는 PyTorch의 가장 핵심적인 개념인 **텐서(Tensor)** 에 대해서 알아보겠습니다.
PyTorch 텐서(Tensor)는 개념적으로 NumPy 배열과 동일합니다:
텐서(Tensor)는 n-차원 배열이며, PyTorch는 이러한 텐서들의 연산을 위한 다양한 기능들을 제공합니다.
NumPy 배열처럼 PyTorch Tensor는 딥러닝이나 연산 그래프, 변화도는 알지 못하며, 과학적 분야의 연산을 위한 포괄적인 도구입니다.
텐서는 연산 그래프와 변화도를 추적할 수도 있지만, 과학적 연산을 위한 일반적인 도구로도 유용합니다.

또한 NumPy와는 다르게, PyTorch 텐서는 GPU를 사용하여 수치 연산을 가속할 수 있습니다.
PyTorch 텐서를 GPU에서 실행하기 위해서는 단지 적절한 장치를 지정해주기만 하면 됩니다.

여기에서는 PyTorch 텐서를 사용하여 3차 다항식을 사인(sine) 함수에 근사해보겠습니다.
위의 NumPy 예제에서와 같이 신경망의 순전파 단계와 역전파 단계는 직접 구현하겠습니다:

.. includenodoc:: /beginner/examples_tensor/polynomial_tensor.py


Autograd
=========

PyTorch: 텐서(Tensor)와 autograd
-------------------------------------------------------------------------------

위의 예제들에서는 신경망의 순전파 단계와 역전파 단계를 직접 구현해보았습니다.
작은 2계층(2-layer) 신경망에서는 역전파 단계를 직접 구현하는 것이 큰일이 아니지만,
복잡한 대규모 신경망에서는 매우 아슬아슬한 일일 것입니다.

다행히도, `자동 미분 <https://en.wikipedia.org/wiki/Automatic_differentiation>`__ 을
사용하여 신경망의 역전파 단계 연산을 자동화할 수 있습니다. PyTorch의 **autograd** 패키지는 정확히
이런 기능을 제공합니다. Autograd를 사용하면, 신경망의 순전파 단계에서 **연산 그래프(computational graph)**
를 정의하게 됩니다; 이 그래프의 노드(node)는 텐서(tensor)이고, 엣지(edge)는 입력 텐서로부터 출력 텐서를
만들어내는 함수가 됩니다. 이 그래프를 통해 역전파를 하게 되면 변화도를 쉽게 계산할 수 있습니다.

이는 복잡하게 들리겠지만, 실제로 사용하는 것은 매우 간단합니다. 각 텐서는 연산그래프에서 노드로 표현됩니다.
만약 ``x`` 가 ``x.requires_grad=True`` 인 텐서라면 ``x.grad`` 어떤 스칼라 값에 대한 ``x`` 의 변화도를 갖는
또 다른 텐서입니다.

여기서는 PyTorch 텐서와 autograd를 사용하여 3차 다항식을 사인파(sine wave)에 근사하는 예제를
구현해보겠습니다; 이제 더 이상 신경망의 역전파 단계를 직접 구현할 필요가 없습니다:

.. includenodoc:: /beginner/examples_autograd/polynomial_autograd.py

PyTorch: 새 autograd Function 정의하기
-------------------------------------------------------------------------------

내부적으로, autograd의 기본(primitive) 연산자는 실제로 텐서를 조작하는 2개의 함수입니다.
**forward** 함수는 입력 텐서로부터 출력 텐서를 계산합니다.
**backward** 함수는 어떤 스칼라 값에 대한 출력 텐서의 변화도(gradient)를 전달받고,
동일한 스칼라 값에 대한 입력 텐서의 변화도를 계산합니다.

PyTorch에서 ``torch.autograd.Function`` 의 하위클래스(subclass)를 정의하고
``forward`` 와 ``backward`` 함수를 구현함으로써 사용자 정의 autograd 연산자를 손쉽게
정의할 수 있습니다. 그 후, 인스턴스(instance)를 생성하고 이를 함수처럼 호출하고,
입력 데이터를 갖는 텐서를 전달하는 식으로 새로운 autograd 연산자를 사용할 수 있습니다.

이 예제에서는 :math:`y=a+bx+cx^2+dx^3` 대신 :math:`y=a+b P_3(c+dx)` 로 모델을
정의합니다. 여기서 :math:`P_3(x)=\frac{1}{2}\left(5x^3-3x\right)` 은 3차
`르장드르 다항식(Legendre polynomial)`_ 입니다. :math:`P_3` 의 순전파와 역전파 연산을
위한 새로운 autograd Function를 작성하고, 이를 사용하여 모델을 구현합니다:

.. _르장드르 다항식(Legendre polynomial):
    https://en.wikipedia.org/wiki/Legendre_polynomials

.. includenodoc:: /beginner/examples_autograd/polynomial_custom_function.py

`nn` 모듈
======================

PyTorch: nn
-------------------------------------------------------------------------------

연산 그래프와 autograd는 복잡한 연산자를 정의하고 도함수(derivative)를 자동으로 계산하는
매우 강력한 패러다임(paradigm)입니다; 하지만 대규모 신경망에서는 autograd 그 자체만으로는 너무
저수준(low-level)일 수 있습니다.

신경망을 구성하는 것을 종종 연산을 **계층(layer)** 에 배열(arrange)하는 것으로 생각하는데,
이 중 일부는 학습 도중 최적화가 될 **학습 가능한 매개변수** 를 갖고 있습니다.

텐서플로우(Tensorflow)에서는, `Keras <https://github.com/fchollet/keras>`__ 와
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
`TFLearn <http://tflearn.org/>`__ 같은 패키지들이 연산 그래프를 고수준(high-level)으로 추상화(abstraction)하여
제공하므로 신경망을 구축하는데 유용합니다.

파이토치(PyTorch)에서는 ``nn`` 패키지가 동일한 목적으로 제공됩니다. ``nn`` 패키지는
신경망 계층(layer)과 거의 비슷한 **Module** 의 집합을 정의합니다. Module은 입력 텐서를 받고
출력 텐서를 계산하는 한편, 학습 가능한 매개변수를 갖는 텐서들을 내부 상태(internal state)로
갖습니다. ``nn`` 패키지는 또한 신경망을 학습시킬 때 주로 사용하는 유용한 손실 함수(loss function)들도
정의하고 있습니다.

이 예제에서는 ``nn`` 패키지를 사용하여 다항식 모델을 구현해보겠습니다:

.. includenodoc:: /beginner/examples_nn/polynomial_nn.py

PyTorch: optim
-------------------------------------------------------------------------------

지금까지는 ``torch.no_grad()`` 로 학습 가능한 매개변수를 갖는 텐서들을 직접 조작하여 모델의 가중치(weight)를 갱신하였습니다.
이것은 확률적 경사하강법(SGD; stochastic gradient descent)와 같은 간단한 최적화 알고리즘에서는 크게 부담이 되지 않지만,
실제로 신경망을 학습할 때는 AdaGrad, RMSProp, Adam 등과 같은 더 정교한 옵티마이저(optimizer)를 사용하곤 합니다.

PyTorch의 ``optim`` 패키지는 최적화 알고리즘에 대한 아이디어를 추상화하고 일반적으로 사용하는 최적화 알고리즘의 구현체(implementation)를
제공합니다.

이 예제에서는 지금까지와 같이 ``nn`` 패키지를 사용하여 모델을 정의하지만, 모델을 최적화할 때는 ``optim`` 패키지가 제공하는
RMSProp 알고리즘을 사용하겠습니다:

.. includenodoc:: /beginner/examples_nn/polynomial_optim.py

PyTorch: 사용자 정의 nn.Module
-------------------------------------------------------------------------------

때대로 기존 Module의 구성(sequence)보다 더 복잡한 모델을 구성해야 할 때가 있습니다;
이러한 경우에는 ``nn.Module`` 의 하위 클래스(subclass)로 새로운 Module을 정의하고,
입력 텐서를 받아 다른 모듈 및 autograd 연산을 사용하여 출력 텐서를 만드는 ``forward`` 를 정의합니다.

이 예제에서는 3차 다항식을 사용자 정의 Module 하위클래스(subclass)로 구현해보겠습니다:

.. includenodoc:: /beginner/examples_nn/polynomial_module.py

PyTorch: 제어 흐름(Control Flow) + 가중치 공유(Weight Sharing)
-------------------------------------------------------------------------------

동적 그래프와 가중치 공유의 예를 보이기 위해, 매우 이상한 모델을 구현해보겠습니다:
각 순전파 단계에서 3 ~ 5 사이의 임의의 숫자(random number)를 선택하여 다차항들에서 사용하고,
동일한 가중치를 여러번 재사용하여 4차항과 5차항을 계산합니다.

이 모델에서는 일반적인 Python 제어 흐름을 사용하여 반복(loop)을 구현할 수 있으며, 순전파 단계를 정의할 때
동일한 매개변수를 여러번 재사용하여 가중치 공유를 구현할 수 있습니다.

이러한 모델을 Module을 상속받는 하위클래스로 간단히 구현해보겠습니다:

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

예제 코드
=============

위의 예제들을 여기서 찾아볼 수 있습니다.

Tensors
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_tensor/polynomial_numpy
   /beginner/examples_tensor/polynomial_tensor

.. galleryitem:: /beginner/examples_tensor/polynomial_numpy.py

.. galleryitem:: /beginner/examples_tensor/polynomial_tensor.py

.. raw:: html

    <div style='clear:both'></div>

Autograd
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_autograd/polynomial_autograd
   /beginner/examples_autograd/polynomial_custom_function


.. galleryitem:: /beginner/examples_autograd/polynomial_autograd.py

.. galleryitem:: /beginner/examples_autograd/polynomial_custom_function.py

.. raw:: html

    <div style='clear:both'></div>

`nn` module
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_nn/polynomial_nn
   /beginner/examples_nn/polynomial_optim
   /beginner/examples_nn/polynomial_module
   /beginner/examples_nn/dynamic_net


.. galleryitem:: /beginner/examples_nn/polynomial_nn.py

.. galleryitem:: /beginner/examples_nn/polynomial_optim.py

.. galleryitem:: /beginner/examples_nn/polynomial_module.py

.. galleryitem:: /beginner/examples_nn/dynamic_net.py

.. raw:: html

    <div style='clear:both'></div>
