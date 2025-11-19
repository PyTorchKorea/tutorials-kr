Compiled Autograd: ``torch.compile`` 을 위해 더 큰 backward 그래프를 포착하기
==========================================================================
**Author:** `Simon Fan <https://github.com/xmfan>`_
**번역:** `이현준 <https://github.com/joonda>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` 무엇을 배울 수 있나요?
       :class-card: card-prerequisites

       * Compiled Autograd가 ``torch.compile`` 와 상호작용하는 방식
       * Compiled Autograd API를 사용하는 방법
       * ``TORCH_LOGS`` 를 사용하여 로그를 검사하는 방법

    .. grid-item-card:: :octicon:`list-unordered;1em;` 전제 조건
       :class-card: card-prerequisites

       * PyTorch 2.4
       * `Introduction to torch.compile <https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html>`_ 완료
       * `Get Started with PyTorch 2.x <https://pytorch.org/get-started/pytorch-2.0/>`_ 의 TorchDynamo와 AOTAutograd 부분을 읽어보세요.

개요
--------
Compiled Autograd는 PyTorch 2.4 에서 소개된  ``torch.compile`` 확장 기능으로, 더 큰 backward 그래프를 캡쳐할 수 있게 해줍니다.

``torch.compile`` 이 backward 그래프를 포작하긴 하지만, 그것은 **부분적으로만** 이루어집니다. AOTAutograd 컴포넌트는 backward 그래프를 사전에 캡쳐하지만, 몇 가지 제한사항이 존재합니다.

* forward 연산에서 그래프 분절이 일어나면, backward 연산에서도 그래프가 분절됩니다.
* `Backward hooks <https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution>`_ 이 캡쳐되지 않습니다

Compiled Autograd는 이러한 제한사항을 해결하기 위해 autograd 엔진과 직접 통합되며, 실행 시 전체 backward 그래프를 캡쳐할 수 있습니다
이러한 두 가지 특성을 가진 모델은 컴파일 Autograd를 시도해보면 좋으며, 잠재적으로 더 좋은 성능을 얻을 수 있습니다.

하지만, Compiled Autograd에도 제한 사항이 존재합니다.

* backward 시작 시 캐시를 확인하기 위해 런타임 오버헤드가 추가됩니다.
* 더 큰 캡쳐를 때문에 dynamo 에서 재컴파일과 그래프 끊김이 발생하기 쉽습니다.

.. note:: Compiled Autograd는 현재 활발히 개발 중이며, 아직 기존 PyTorch 기능과 완전히 호환되지 않습니다. 특정 기능의 최신 상태는 `Compiled Autograd Landing Page <https://docs.google.com/document/d/11VucFBEewzqgkABIjebZIzMvrXr3BtcY1aGKpX61pJY>`_ 를 참고하세요

설정
-----
이 튜토리얼에서, 간단한 신경망 모델을 기반으로 예제를 진행합니다.
10차원의 입력 벡터를 받아, 단일 선형 레이어를 통과시킨 후, 또 다른 10차원 출력 벡터를 생성합니다.

.. code:: python

   import torch

   class Model(torch.nn.Module):
      def __init__(self):
         super().__init__()
         self.linear = torch.nn.Linear(10, 10)

      def forward(self, x):
         return self.linear(x)

기본 사용
------------
``torch.compile`` API를 호출 하기 전에, ``torch._dynamo.config.compiled_autograd`` 을 ``True`` 로 설정해주세요

.. code:: python

   model = Model()
   x = torch.randn(10)

   torch._dynamo.config.compiled_autograd = True
   @torch.compile
   def train(model, x):
      loss = model(x).sum()
      loss.backward()

   train(model, x)

위에 있는 코드는, ``Model`` 클래스 인스턴스를 만들고 ``torch.randn(10)`` 을 사용하여 무작위 10차원의 텐서 ``x`` 를 만듭니다.
훈련 루프 함수 ``train`` 을 정의하고, 실행 최적화를 위해 @torch.compile로 지정합니다.
``train(model, x)`` 가 호출될 때:

* Python 인터프리터가 Dynamo를 호출합니다. 해당 호출은 ``@torch.compile`` 로 지정되었기 때문입니다.
* Dynamo는 Python 바이트코드를 가로채 실행을 시뮬레이션하고, 연산을 그래프로 기록합니다.
* ``AOTDispatcher`` 은 훅을 비활성화하고 autograd 엔진을 호출하여 ``model.linear.weight`` 와 ``model.linear.bias`` 의 변화도를 계산하며, 연산을 그래프로 기록합니다. ``torch.autograd.Function`` 을 사용하여, AOTDispatcher는 ``train`` 함수의 forward와 backward 구현을 재작성합니다.
* Inductor는 AOTDispatch forward와 backward 구현을 최적화한 함수를 생성합니다.
* Dynamo는 Python 인터프리터가 다음에 평가할 최적화된 함수를 설정합니다.
* Python 인터프리터는 최적화된 함수를 실행하고, ``loss = model(x).sum()`` 을 실행합니다.
* Python 인터프리터는 ``loss.backward()`` 를 실행하며 내부 autograd 엔진을 호출하고, ``torch._dynamo.config.compiled_autograd = True`` 로 설정했기 때문에, 해당 호출은 Compiled Autograd 엔진으로 전달됩니다.
* Compiled Autograd 는 ``model.linear.weight`` 와 ``model.linear.bias`` 변화도를 계산하고, 만나는 훅을 포함하여 연산을 그래프로 기록합니다. 이 과정에서 AOTDispatcher가 이전에 재작성한 backward도 기록됩니다. 그 다음 Compiled Autograd는 ``loss.backward()`` 의 완전히 추적된 구현에 해당하는 새 함수를 생성하고, 이를 추론 모드에서 ``torch.compile`` 로 실행합니다.
* 동일한 단계에서 재귀적으로 Compiled Autograd 그래프에 적용되지만, 이번에는 AOTDispatcher가 그래프를 분할할 필요가 없습니다.

Compiled Autograd 로그 검사
-------------------------------------
``TORCH_LOGS`` 환경 변수를 설정하여 스크립트를 실행합니다:

* Compiled Autograd 그래프를 출력하려면 ``TORCH_LOGS="compiled_autograd" python example.py`` 을 사용하세요
* 더 많은 텐서 메타데이터와 재컴파일 이유까지 출력하고 싶다면, 성능이 저하되는 대신 ``TORCH_LOGS="compiled_autograd_verbose" python example.py`` 를 사용하세요

위의 스피넷을 다시 실행하면, compiled autograd 그래프가 ``stderr`` 에 로깅이 됩니다.
일부 그래프 노드는 ``aot0_`` 접두사가 붙은 이름을 가지며, 이는 이전에 AOTAutograd backward 그래프 0에서 사전 컴파일된 노드에 해당합니다, 예를 들어 ``aot0_view_2`` 는 id=0인 AOT backward 그래프의 ``view_2`` 에 대응됩니다.
아래의 이미지에서, 빨간 박스는 Compiled Autograd 없이 ``torch.compile`` 로 캡쳐된 AOT backward 그래프를 감싸고 있습니다.


.. image:: ../_static/img/compiled_autograd/entire_verbose_log.png

.. note:: 이 그래프는 우리가 ``torch.compile`` 을 호출할 대상이며, 최적화된 그래프가 **아닙니다.** Compiled Autograd는 기본적으로 C++ autograd 실행을 나타내기 위해 일부 최적화되지 않은 Python 코드를 생성합니다.

다른 플래그를 사용하여 forward와 backward 패스를 컴파일하기
-------------------------------------------------------------
두 가지의 컴파일에 대해 서로 다른 컴파일러 설정을 사용할 수 있습니다, 예를 들어 forward에 그래프 분절이 있더라도 backward는 fullgraph로 설정할 수 있습니다.

.. code:: python

   def train(model, x):
       model = torch.compile(model)
       loss = model(x).sum()
       torch._dynamo.config.compiled_autograd = True
       torch.compile(lambda: loss.backward(), fullgraph=True)()

또는 context manager를 사용할 수 있으며, 해당 스코프 안의 모든 autograd 호출에 적용될 것이다.

.. code:: python

   def train(model, x):
      model = torch.compile(model)
      loss = model(x).sum()
      with torch._dynamo.compiled_autograd.enable(torch.compile(fullgraph=True)):
         loss.backward()


Compiled Autograd는 AOTAutograd의 특정 한계점을 해결합니다.
--------------------------------------------------------------
1. forward 패스의 그래프 분절은 더 이상 backward 패스의 그래프 분절로 이어지지 않습니다.

.. code:: python

   @torch.compile(backend="aot_eager")
   def fn(x):
      # 1st graph
      temp = x + 10
      torch._dynamo.graph_break()
      # 2nd graph
      temp = temp + 10
      torch._dynamo.graph_break()
      # 3rd graph
      return temp.sum()

   x = torch.randn(10, 10, requires_grad=True)
   torch._dynamo.utils.counters.clear()
   loss = fn(x)

   # 1. base torch.compile
   loss.backward(retain_graph=True)
   assert(torch._dynamo.utils.counters["stats"]["unique_graphs"] == 3)
   torch._dynamo.utils.counters.clear()

   # 2. torch.compile with compiled autograd
   with torch._dynamo.compiled_autograd.enable(torch.compile(backend="aot_eager")):
      loss.backward()

   # single graph for the backward
   assert(torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1)


첫 번째 ``torch.compile`` 의 경우에는, 컴파일된 함수 ``fn`` 에서 2개의 그래프 분절로 인해 3개의 backward 그래프가 생성된 것을 확인할 수 있습니다.
반면, Compiled Autograd를 사용한 두 번째 ``torch.compile`` 경우에는 그래프 분절이 있더라도 전체 backward 그래프가 트레이스된 것을 확인할 수 있습니다.

.. note:: Compiled Autograd가 캡쳐한 backward 훅을 트레이스할 때, Dynamo에서 그래프가 분절될 가능성은 여전히 존재합니다.


2. Backward 훅은 캡쳐될 수 있습니다.

.. code:: python

   @torch.compile(backend="aot_eager")
   def fn(x):
      return x.sum()

   x = torch.randn(10, 10, requires_grad=True)
   x.register_hook(lambda grad: grad+10)
   loss = fn(x)

   with torch._dynamo.compiled_autograd.enable(torch.compile(backend="aot_eager")):
      loss.backward()

그래프에는 ``call_hook`` 노드가 있어야 하며, 이후 dynamo는 이를 다음과 같이 인라인 처리합니다.

.. image:: ../_static/img/compiled_autograd/call_hook_node.png

Compiled Autograd의 공통적인 재컴파일 이유
--------------------------------------------------
1. 손실 값의 autograd 구조가 변경되었기 때문입니다.

.. code:: python

   torch._dynamo.config.compiled_autograd = True
   x = torch.randn(10, requires_grad=True)
   for op in [torch.add, torch.sub, torch.mul, torch.div]:
      loss = op(x, x).sum()
      torch.compile(lambda: loss.backward(), backend="eager")()

위의 예제에서, 각 반복마다 다른 연산을 호출하여 ``loss`` 가 매번 다른 autograd 기록을 추적합니다. 이로 인해 재컴파일 메시지가 (**Cache miss due to new autograd node**) 표시되는 것을 확인할 수 있습니다.

.. image:: ../_static/img/compiled_autograd/recompile_due_to_node.png

2. 텐서의 형태가 변경되었기 때문입니다.

.. code:: python

   torch._dynamo.config.compiled_autograd = True
   for i in [10, 100, 10]:
      x = torch.randn(i, i, requires_grad=True)
      loss = x.sum()
      torch.compile(lambda: loss.backward(), backend="eager")()

위의 예제에서, ``x`` 의 형태가 변경되면, compiled autograd는 첫 번째 변경 이후 ``x`` 를 동적 형태 텐서로 표시합니다. 이로 인해 재컴파일 메시지가 (**Cache miss due to changed shapes**) 나타나는 것을 확인할 수 있습니다.

.. image:: ../_static/img/compiled_autograd/recompile_due_to_dynamic.png

결론
----------
이 튜토리얼에서는, ``torch.compile`` 과 compiled autograd의 고차원 생태계, compiled autograd의 기초와 몇 가지의 공통적인 재컴파일 이유를 살펴보았습니다. 자세한 내용은 `dev-discuss <https://dev-discuss.pytorch.org/>`_ 에서 확인할 수 있습니다.
