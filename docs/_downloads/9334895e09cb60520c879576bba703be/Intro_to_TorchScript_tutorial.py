"""
TorchScript 소개
===========================

**Author**: James Reed (jamesreed@fb.com), Michael Suo (suo@fb.com), rev2

**번역**: `강준혁 <https://github.com/k1101jh>`_

이 튜토리얼은 C++와 같은 고성능 환경에서 실행될 수 있는
PyTorch 모델(``nn.Module`` 의 하위클래스)의 중간 표현인
TorchScript에 대한 소개입니다.

이 튜토리얼에서는 다음을 다룰 것입니다:

1. 다음을 포함한 PyTorch의 모델 제작의 기본:

-  모듈(Modules)
-  ``forward`` 함수 정의하기
-  모듈을 계층 구조로 구성하기

2. PyTorch 모듈을 고성능 배포 런타임인 TorchScript로 변환하는 특정 방법

-  기존 모듈 추적하기
-  스크립트를 사용하여 모듈을 직접 컴파일하기
-  두 가지 접근 방법을 구성하는 방법
-  TorchScript 모듈 저장 및 불러오기

이 튜토리얼을 완료한 후에는
`다음 학습서 <https://tutorials.pytorch.kr/advanced/cpp_export.html>`_
를 통해 C++에서 TorchScript 모델을 실제로 호출하는 예제를 안내합니다.

"""

import torch  # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)


######################################################################
# PyTorch 모델 작성의 기초
# ---------------------------------
#
# 간단한 ``Module`` 을 정의하는 것부터 시작하겠습니다. ``Module`` 은 PyTorch의
# 기본 구성 단위입니다. 이것은 다음을 포함합니다:
#
# 1. 호출을 위해 모듈을 준비하는 생성자
# 2. ``Parameters`` 집합과 하위 ``Module`` . 이것들은 생성자에 의해 초기화되며
#    호출 중에 모듈에 의해 사용될 수 있습니다.
# 3. ``forward`` 함수. 모듈이 호출될 때 실행되는 코드입니다.
#
# 작은 예제로 시작해 보겠습니다:
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h

my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))


######################################################################
# 우리는 다음 작업을 수행했습니다.:
#
# 1. 하위 클래스로 ``torch.nn.Module`` 을 갖는 클래스를 생성했습니다.
# 2. 생성자를 정의했습니다. 생성자는 많은 작업을 수행하지 않고 ``super`` 로
#    생성자를 호출합니다.
# 3. 두 개의 입력을 받아 두 개의 출력을 반환하는 ``forward`` 함수를 정의했습니다.
#    ``forward`` 함수의 실제 내용은 크게 중요하진 않지만, 가짜 `RNN
#    cell <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__ 의
#    일종입니다. 즉, 반복(loop)에 적용되는 함수입니다.
#
# 모듈을 인스턴스화하고, 3x4 크기의 무작위 값들로 이루어진 행렬 ``x`` 와 ``h`` 를
# 만들었습니다.
# 그런 다음, ``my_cell(x, h)`` 를 이용해 cell을 호출했습니다. 이것은 ``forward``
# 함수를 호출합니다.
#
# 좀 더 흥미로운 것을 해봅시다:
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))


######################################################################
# 모듈 ``MyCell`` 을 재정의했지만, 이번에는 ``self.linear`` 속성을 추가하고
# forward 함수에서 ``self.linear`` 을 호출했습니다.
#
# 여기서 무슨 일이 일어날까요? ``torch.nn.Linear`` 은 ``MyCell`` 과
# 마찬가지로 PyTorch 표준 라이브러리의 ``Module`` 입니다. 이것은 호출 구문을
# 사용하여 호출할 수 있습니다. 우리는 ``Module`` 의 계층을 구축하고 있습니다.
#
# ``Module`` 에서 ``print`` 하는 것은 ``Module`` 의 하위 클래스 계층에 대한
# 시각적 표현을 제공할 것입니다. 이 예제에서는 ``Linear`` 의 하위 클래스와
# 하위 클래스의 매개 변수를 볼 수 있습니다.
#
# ``Module`` 을 이런 방식으로 작성하면, 재사용 가능한 구성 요소를 사용하여
# 모델을 간결하고 읽기 쉽게 작성할 수 있습니다.
#
# 여러분은 출력된 내용에서 ``grad_fn`` 을 확인하셨을 것입니다. 이것은
# `오토그라드(autograd) <https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html>`__
# 라 불리는 PyTorch의 자동 미분 방법의 세부 정보입니다. 요컨데, 이 시스템은
# 잠재적으로 복잡한 프로그램을 통해 미분을 계산할 수 있게 합니다. 이 디자인은
# 모델 제작에 엄청난 유연성을 제공합니다.
#
# 이제 유연성을 시험해 보겠습니다.
#

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))


######################################################################
# MyCell 클래스를 다시 정의했지만, 여기선 ``MyDecisionGate`` 를 정의했습니다.
# 이 모듈은 **제어 흐름** 을 활용합니다. 제어 흐름은 루프와 ``if`` 명령문과
# 같은 것으로 구성됩니다.
#
# 많은 프레임워크들은 주어진 프로그램 코드로부터 기호식 미분(symbolic
# derivatives)을 계산하는 접근법을 취하고 있습니다. 하지만, PyTorch에서는 변화도
# 테이프(gradient tape)를 사용합니다. 연산이 발생할 때 이를 기록하고, 미분값을
# 계산할 때 거꾸로 재생합니다. 이런 방식으로, 프레임워크는 언어의 모든 구문에
# 대한 미분값을 명시적으로 정의할 필요가 없습니다.
#
# .. figure:: https://github.com/pytorch/pytorch/raw/master/docs/source/_static/img/dynamic_graph.gif
#    :alt: 오토그라드가 작동하는 방식
#
#    오토그라드가 작동하는 방식
#


######################################################################
# TorchScript의 기초
# ---------------------
#
# 이제 실행 예제를 살펴보고 TorchScript를 적용하는 방법을 살펴보겠습니다.
#
# 한마디로, TorchScript는 PyTorch의 유연하고 동적인 특성을 고려하여 모델 정의를
# 캡쳐할 수 있는 도구를 제공합니다.
# **추적(tracing)** 이라 부르는 것을 검사하는 것으로 시작하겠습니다.
#
# ``Module`` 추적
# ~~~~~~~~~~~~~~~~~~~
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)


######################################################################
# 살짝 앞으로 돌아가 ``MyCell`` 의 두 번째 버전을 가져왔습니다. 이전에 이것을
# 인스턴스화 했지만 이번엔 ``torch.jit.trace`` 를 호출하고, ``Module`` 을
# 전달했으며, 네트워크가 볼 수 있는 *입력 예* 를 전달했습니다.
#
# 여기서 무슨 일이 발생했습니까? ``Module`` 을 호출하였고, ``Module`` 이 돌아갈 때
# 발생한 연산을 기록하였고, ``torch.jit.ScriptModule`` 의 인스터스를 생성했습니다.
# ( ``TracedModule`` 은 인스턴스입니다)
#
# TorchScript는 일반적으로 딥 러닝에서 *그래프* 라고 하는 중간 표현(또는 IR)에
# 정의를 기록합니다. ``.graph`` 속성으로 그래프를 확인해볼 수 있습니다:
#

print(traced_cell.graph)


######################################################################
# 그러나, 이것은 저수준의 표현이며 그래프에 포함된 대부분의 정보는
# 최종 사용자에게 유용하지 않습니다. 대신, ``.code`` 속성을 사용하여 코드에
# 대한 Python 구문 해석을 제공할 수 있습니다:
#

print(traced_cell.code)


######################################################################
# **어째서** 이런 일들을 했을까요? 여기에는 몇 가지 이유가 있습니다:
#
# 1. TorchScript 코드는 기본적으로 제한된 Python 인터프리터인 자체 인터프리터에서
#    호출될 수 있습니다. 이 인터프리터는 GIL(Global Interpreter Lock)을 얻지
#    않으므로 동일한 인스턴스에서 동시에 많은 요청을 처리할 수 있습니다.
# 2. 이 형식을 사용하면 전체 모델을 디스크에 저장하고 Python 이외의 언어로 작성된
#    서버와 같은 다른 환경에서 불러올 수 있습니다.
# 3. TorchScript는 보다 효율적인 실행을 제공하기 위해 코드에서 컴파일러 최적화를
#    수행할 수 있는 표현을 제공합니다.
# 4. TorchScript를 사용하면 개별 연산자보다 프로그램의 더 넓은 관점을 요구하는 많은
#    백엔드/장치 런타임과 상호작용(interface)할 수 있습니다.
#
# ``traced_cell`` 을 호출하면 Python 모듈과 동일한 결과가 생성됩니다:
#

print(my_cell(x, h))
print(traced_cell(x, h))


######################################################################
# 스크립팅을 사용하여 모듈 변환
# ----------------------------------
#
# 제어 흐름이 포함된(control-flow-laden) 하위 모듈이 아닌 모듈 버전 2를 사용하는
# 이유가 있습니다. 지금 살펴봅시다:
#

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)


######################################################################
# ``.code`` 출력을 보면, ``if-else`` 분기가 어디에도 없다는 것을 알 수 있습니다!
# 어째서일까요? 추적은 코드를 실행하고 *발생하는* 작업을 기록하며 정확하게 수행하는
# 스크립트 모듈(ScriptModule)을 구성하는 일을 수행합니다. 불행하게도, 제어 흐름과
# 같은 것들은 지워집니다.
#
# TorchScript에서 이 모듈을 어떻게 충실하게 나타낼 수 있을까요? Python 소스 코드를
# 직접 분석하여 TorchScript로 변환하는 **스크립트 컴파일러(script compiler)** 를
# 제공합니다. ``MyDecisionGate`` 를 스크립트 컴파일러를 사용하여 변환해 봅시다:
#

scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print(scripted_gate.code)
print(scripted_cell.code)


######################################################################
# 만세! 이제 TorchScript에서 프로그램의 동작을 충실하게 캡쳐했습니다. 이제
# 프로그램을 실행해 봅시다:
#

# 새로운 입력
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell(x, h)


######################################################################
# 스크립팅과 추적 혼합
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 어떤 상황에서는 스크립팅보다는 추적을 사용해야 합니다. (예: 모듈에는 TorchScript에
# 표시하지 않으려는 Python 상수 값을 기반으로 만들어진 많은 구조적인
# 결정(architectural decisions)이 있습니다.) 이 경우, 스크립팅은 추적으로
# 구성될 수 있습니다: ``torch.jit.script`` 는 추적된 모듈의 코드를 인라인(inline)
# 할 것이고, 추적은 스크립트 된 모듈의 코드를 인라인 할 것입니다.
#
# 첫 번째 경우의 예:
#

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)



######################################################################
# 두 번째 경우의 예:
#

class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)


######################################################################
# 이러한 방식으로, 스크립팅과 추적은 상황에 따라서 따로 사용되거나, 함께
# 사용될 수 있습니다.
#
# 모델 저장 및 불러오기
# -------------------------
#
# TorchScript 모듈을 아카이브 형식으로 디스크에 저장하고 불러오는 API를 제공합니다.
# 이 형식은 코드, 매개 변수, 속성과 디버그 정보를 포함합니다. 이것은 그 아카이브가
# 완전히 별개의 프로세스로 로드할 수 있는 모델의 독립 표현임을 의미합니다.
# 랩핑 된 RNN 모듈을 저장하고 로드해 봅시다:
#

traced.save('wrapped_rnn.pt')

loaded = torch.jit.load('wrapped_rnn.pt')

print(loaded)
print(loaded.code)


######################################################################
# 보시다시피, 직렬화는 모듈 계층과 검사한 코드를 유지합니다. 또한 모델을 로드할
# 수 있습니다. 예를 들어, Python 없이 실행하기 위해 모델을
# `C++ <https://tutorials.pytorch.kr/advanced/cpp_export.html>`__ 로 로드할
# 수 있습니다.
#
# 더 읽을거리
# ~~~~~~~~~~~~~~~
# 튜토리얼을 완료했습니다! 관련 데모를 보려면 TorchScript를 사용하여 기계 번역
# 모델을 변환하기 위한 NeurIPS 데모를 확인하십시오:
# https://colab.research.google.com/drive/1HiICg6jRkBnr5hvK2-VnMi88Vi9pUzEJ
#
