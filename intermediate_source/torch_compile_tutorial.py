# -*- coding: utf-8 -*-

"""
``torch.compile`` 소개
=================================
**저자**: William Wen
**번역**: `명준현 <https://github.com/Junhyun17>`_
"""

######################################################################
# ``torch.compile``은 PyTorch 코드를 더 빠르게 실행하는 최신 메소드입니다!
# ``torch.compile``은 PyTorch 코드를
# 최적화된 커널로 JIT 컴파일하여
# 코드 변경을 최소화하면서 PyTorch 코드를 더 빠르게 실행합니다.
#
# 이 튜토리얼에서는 ``torch.compile``의 기본 사용법을 다루며
# `TorchScript <https://pytorch.org/docs/stable/jit.html>`__ 및
# `FX Tracing <https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace>`__\ 과 같은
# 이전 PyTorch 컴파일러 솔루션에 비해
# ``torch.compile``의 장점을 보여줍니다.
#
# **목차**
#
# .. contents::
#     :local:
#
# **필수 pip 의존성**
#
# - ``torch >= 2.0``
# - ``torchvision``
# - ``numpy``
# - ``scipy``
# - ``tabulate``
#
# **시스템 요구 사항**
# - ``g++``와 같은 C++ 컴파일러
# - Python 개발 패키지(``python-devel``/``python-dev``)

######################################################################
# 참고: 아래 및 다른 문서에 제시된 속도 향상 수치를 재현하기 위해
# 이 튜토리얼에는 최신 NVIDIA GPU(H100, A100 또는 V100)를 사용하는 것이 좋습니다.

import torch
import warnings

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )

######################################################################
# 기본 사용법
# ------------
#
# ``torch.compile``은 최신 PyTorch에 포함되어 있습니다.
# GPU에서 TorchInductor를 실행하려면 Triton이 필요하며 Triton은 PyTorch 2.0 nightly
# 바이너리에 포함되어 있습니다. Triton이 없다면 pip로 ``torchtriton`` 설치를 시도해보세요
# (CUDA 11.7의 경우
# ``pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu117"``).
#
# 임의의 Python 함수는 호출 가능한 객체(callable)를
# ``torch.compile``에 전달하여 최적화할 수 있습니다.
# 그러면 반환된 최적화 함수를 원래 함수 대신 호출할 수 있습니다.

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

######################################################################
# 또는 함수에 데코레이터를 사용할 수 있습니다.
t1 = torch.randn(10, 10)
t2 = torch.randn(10, 10)

@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
print(opt_foo2(t1, t2))

######################################################################
# ``torch.nn.Module`` 인스턴스도 최적화할 수 있습니다.

t = torch.randn(10, 100)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule()
mod.compile()
print(mod(t))
## 또는:
# opt_mod = torch.compile(mod)
# print(opt_mod(t))

######################################################################
# torch.compile과 중첩 호출
# ------------------------------
# ``torch.compile``은 데코레이트한 함수 안의 중첩 함수 호출도 함께 컴파일합니다.

def nested_function(x):
    return torch.sin(x)

@torch.compile
def outer_function(x, y):
    a = nested_function(x)
    b = torch.cos(y)
    return a + b

print(outer_function(t1, t2))

######################################################################
# 같은 방식으로 모듈을 컴파일할 때 컴파일에서 제외할 목록(skip list)에 없는
# 모듈 안의 모든 하위 모듈과 메소드도 함께 컴파일합니다.

class OuterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_module = MyModule()
        self.outer_lin = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = self.inner_module(x)
        return torch.nn.functional.relu(self.outer_lin(x))

outer_mod = OuterModule()
outer_mod.compile()
print(outer_mod(t))

######################################################################
# ``torch.compiler.disable``을 사용하여 일부 함수의 컴파일을 비활성화할 수도 있습니다.
# ``complex_function`` 함수에서만 추적을 비활성화하고
# ``complex_conjugate``에서는 다시 추적을 계속하고 싶다고 가정해봅시다.
# 이 경우 ``torch.compiler.disable(recursive=False)`` 옵션을 사용할 수 있습니다.
# 그렇지 않으면 기본값은 ``recursive=True``입니다.

def complex_conjugate(z):
    return torch.conj(z)

@torch.compiler.disable(recursive=False)
def complex_function(real, imag):
    # 이 함수가 컴파일 중 문제를 일으킨다고 가정합니다.
    z = torch.complex(real, imag)
    return complex_conjugate(z)

def outer_function():
    real = torch.tensor([2, 3], dtype=torch.float32)
    imag = torch.tensor([4, 5], dtype=torch.float32)
    z = complex_function(real, imag)
    return torch.abs(z)

# outer_function 컴파일을 시도합니다.
try:
    opt_outer_function = torch.compile(outer_function)
    print(opt_outer_function())
except Exception as e:
    print("Compilation of outer_function failed:", e)

######################################################################
# 모범 사례와 권장 사항
# ----------------------------------
#
# 중첩 모듈과 함수 호출에서 ``torch.compile``의 동작
#
# ``torch.compile``을 사용하면 컴파일러는 대상 함수 또는 모듈 안에서 호출되는
# 모든 함수 중 컴파일에서 제외할 목록(skip list)에 없는 함수를 재귀적으로
# 컴파일하려고 시도합니다(예: 내장 함수, torch.* 네임스페이스의 일부 함수).
#
# **모범 사례**
#
# 1. **최상위 수준 컴파일** 한 가지 방법은 가능한 가장 높은 수준에서
# 컴파일하고(즉, 최상위 모듈을 초기화하거나 호출할 때) 과도한 그래프 분리나
# 오류가 발생하면 선택적으로 컴파일을 비활성화하는 것입니다.
# 그래도 컴파일 문제가 많이 남아 있다면
# 대신 개별 하위 구성 요소를 컴파일합니다.
#
# 2. **모듈식 테스트** 큰 모델에 통합하기 전에 개별 함수와 모듈을
# ``torch.compile``로 테스트하여 잠재적인 문제를 분리합니다.
#
# 3. **선택적으로 컴파일 비활성화** 특정 함수나 하위 모듈을 ``torch.compile``에서
# 처리할 수 없다면 ``torch.compiler.disable`` 컨텍스트 매니저를 사용하여
# 해당 함수나 하위 모듈을 컴파일에서 재귀적으로 제외합니다.
#
# 4. **리프 함수 먼저 컴파일** 중첩 함수와 모듈이 여러 개 있는 복잡한 모델에서는
# 리프 함수나 모듈부터 먼저 컴파일합니다. 자세한 내용은
# `세밀한 추적을 위한 TorchDynamo API <https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html>`__\ 를 참고하세요.
#
# 5. **``torch.compile(mod)``보다 ``mod.compile()`` 선호** ``state_dict``에서
# ``_orig_`` 접두사 문제가 발생하지 않도록 합니다.
#
# 6. **그래프 분리를 잡기 위해 ``fullgraph=True`` 사용** 종단 간 컴파일을 보장하도록
# 도와 속도 향상을 극대화하고 ``torch.export``와의 호환성을 높입니다.


######################################################################
# 속도 향상 확인하기
# -----------------------
#
# 이제 ``torch.compile``을 사용하면 실제 모델의 속도를 높일 수 있음을 보여줍니다.
# 무작위 데이터에서 ``torchvision`` 모델을 평가하고 학습하여 표준 즉시 실행 모드와
# ``torch.compile``을 비교합니다.
#
# 시작하기 전에 몇 가지 유틸리티 함수를 정의해야 합니다.

# `fn()`\ 을 실행한 결과와 `fn()` 실행에 걸린 시간을 초 단위로 반환합니다.
# 가장 정확하게 측정하기 위해
# CUDA 이벤트와 동기화를 사용합니다.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# 모델에 사용할 무작위 입력과 대상 데이터를 생성하며 여기서 `b`\ 는
# 배치 크기입니다.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).cuda()

######################################################################
# 먼저 추론을 비교해봅시다.
#
# ``torch.compile``을 호출할 때 추가 ``mode`` 인자가 있으며
# 이에 대해서는 아래에서 설명합니다.

model = init_model()

# 다른 모드를 사용하므로 초기화합니다.
import torch._dynamo
torch._dynamo.reset()

model_opt = torch.compile(model, mode="reduce-overhead")

inp = generate_data(16)[0]
with torch.no_grad():
    print("eager:", timed(lambda: model(inp))[1])
    print("compile:", timed(lambda: model_opt(inp))[1])

######################################################################
# ``torch.compile``은 eager와 비교해 완료하는 데 훨씬 더 오래 걸린다는 점에
# 주목하세요. 이는 ``torch.compile``이 실행 중에 모델을 최적화된 커널로 컴파일하기 때문입니다.
# 이 예제에서는 모델의 구조가 바뀌지 않으므로 다시 컴파일할 필요가 없습니다.
# 따라서 최적화한 모델을 몇 번 더 실행하면 eager와 비교해
# 상당한 개선을 확인할 수 있습니다.

eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    with torch.no_grad():
        _, eager_time = timed(lambda: model(inp))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    with torch.no_grad():
        _, compile_time = timed(lambda: model_opt(inp))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
print("~" * 10)

import numpy as np
eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert(speedup > 1)
print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)

######################################################################
# 실제로 ``torch.compile``로 모델을 실행하면 상당한 속도 향상이 나타나는 것을 확인할 수 있습니다.
# 속도 향상은 주로 Python 오버헤드와 GPU 읽기·쓰기를 줄이는 데서 나오므로
# 관찰되는 속도 향상은 모델 구조와 배치 크기 같은 요인에 따라 달라질 수 있습니다.
# 예를 들어 모델 구조가 단순하고 데이터의 양이 많다면 GPU 연산에서 병목이 발생하며
# 관찰되는 속도 향상이 크지 않을 수 있습니다.
#
# 선택한 ``mode`` 인자에 따라 다른 속도 향상 결과를 볼 수도 있습니다.
# ``"reduce-overhead"`` 모드는 CUDA 그래프를 사용하여 Python 오버헤드를
# 더 줄입니다. 직접 만든 모델에서는 속도 향상을 극대화하기 위해
# 여러 모드를 실험해야 할 수도 있습니다. 모드에 대한 자세한 내용은
# `여기 <https://pytorch.org/get-started/pytorch-2.0/#user-experience>`__\ 에서 읽을 수 있습니다.
#
# 또한 ``torch.compile``로 모델을 두 번째로 실행할 때 첫 번째 실행보다는 훨씬 빠르지만
# 다른 실행보다 상당히 느리다는 점을 볼 수도 있습니다. 이는 ``"reduce-overhead"``
# 모드가 CUDA 그래프를 위해 몇 번의 워밍업 반복을 실행하기 때문입니다.
#
# 일반적인 PyTorch 벤치마킹에는 위에서 정의한 ``timed`` 함수 대신
# ``torch.utils.benchmark``를 사용할 수 있습니다. 이 튜토리얼에서는
# ``torch.compile``의 컴파일 지연 시간을 보여주기 위해 자체 시간 측정 함수를 작성했습니다.
#
# 이제 학습을 비교해보겠습니다.

model = init_model()
opt = torch.optim.Adam(model.parameters())

def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, eager_time = timed(lambda: train(model, inp))
    eager_times.append(eager_time)
    print(f"eager train time {i}: {eager_time}")
print("~" * 10)

model = init_model()
opt = torch.optim.Adam(model.parameters())
train_opt = torch.compile(train, mode="reduce-overhead")

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, compile_time = timed(lambda: train_opt(model, inp))
    compile_times.append(compile_time)
    print(f"compile train time {i}: {compile_time}")
print("~" * 10)

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert(speedup > 1)
print(f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)

######################################################################
# 마찬가지로 ``torch.compile``은 모델을 컴파일해야 하므로 첫 번째 반복에서는
# 더 오래 걸리지만, 이후 반복에서는 eager와 비교해
# 상당한 속도 향상을 확인할 수 있습니다.
#
# 이 튜토리얼에 제시된 속도 향상 수치는 설명을 위한 예시일 뿐입니다.
# 공식 속도 향상 수치는
# `TorchInductor 성능 대시보드 <https://hud.pytorch.org/benchmark/compilers>`__\ 에서 확인할 수 있습니다.

######################################################################
# TorchScript 및 FX Tracing과 비교
# -----------------------------------------
#
# 지금까지 ``torch.compile``이 PyTorch 코드의 속도를 높일 수 있음을 살펴봤습니다.
# 그렇다면 TorchScript나 FX Tracing과 같은 기존 PyTorch 컴파일러 솔루션보다
# ``torch.compile``을 사용해야 하는 또 다른 이유는 무엇일까요? 주된 장점은
# ``torch.compile``이 기존 코드를 최소한만 변경하여 임의의 Python 코드를
# 처리할 수 있다는 점에 있습니다.
#
# 다른 컴파일러 솔루션은 처리하기 어려워하지만 ``torch.compile``은 처리할 수 있는
# 한 가지 경우가 데이터 의존 제어 흐름(data-dependent control flow)입니다
# (아래의 ``if x.sum() < 0:`` 줄).

def f1(x, y):
    if x.sum() < 0:
        return -y
    return y

# 동일한 인자 `args`\ 가 주어졌을 때 `fn1`\ 과 `fn2`\ 가 같은 결과를 반환하는지 테스트합니다.
# 일반적으로 `fn1`\ 은 eager 함수이고 `fn2`\ 는 컴파일된 함수입니다
# (torch.compile, TorchScript 또는 FX graph).
def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)

######################################################################
# TorchScript로 ``f1``을 추적하면 실제 제어 흐름 경로만 추적하므로
# 경고 없이 잘못된 결과가 나옵니다.

traced_f1 = torch.jit.trace(f1, (inp1, inp2))
print("traced 1, 1:", test_fns(f1, traced_f1, (inp1, inp2)))
print("traced 1, 2:", test_fns(f1, traced_f1, (-inp1, inp2)))

######################################################################
# FX 추적은 데이터 의존 제어 흐름이 있으므로
# ``f1``에서 오류를 발생시킵니다.

import traceback as tb
try:
    torch.fx.symbolic_trace(f1)
except:
    tb.print_exc()

######################################################################
# FX로 ``f1``을 추적하려고 할 때 ``x`` 값을 제공하면 추적된 함수에서
# 데이터 의존 제어 흐름이 제거되므로 TorchScript 추적과 같은 문제가 발생합니다.

fx_f1 = torch.fx.symbolic_trace(f1, concrete_args={"x": inp1})
print("fx 1, 1:", test_fns(f1, fx_f1, (inp1, inp2)))
print("fx 1, 2:", test_fns(f1, fx_f1, (-inp1, inp2)))

######################################################################
# 이제 ``torch.compile``이 데이터 의존 제어 흐름을 올바르게 처리하는 것을
# 확인할 수 있습니다.

# 다른 모드를 사용하므로 초기화합니다.
torch._dynamo.reset()

compile_f1 = torch.compile(f1)
print("compile 1, 1:", test_fns(f1, compile_f1, (inp1, inp2)))
print("compile 1, 2:", test_fns(f1, compile_f1, (-inp1, inp2)))
print("~" * 10)

######################################################################
# TorchScript scripting은 데이터 의존 제어 흐름을 처리할 수 있지만
# 이 솔루션에는 자체적인 문제가 따릅니다. 구체적으로 TorchScript scripting은
# 코드를 크게 변경해야 할 수 있으며 지원하지 않는 Python 기능을 사용하면
# 오류를 발생시킵니다.
#
# 아래 예제에서는 TorchScript 타입 주석을 잊어버렸고, 인자 ``y``의 입력 타입인 ``int``가
# 기본 인자 타입인 ``torch.Tensor``와 일치하지 않기 때문에
# TorchScript 오류가 발생합니다.

def f2(x, y):
    return x + y

inp1 = torch.randn(5, 5)
inp2 = 3

script_f2 = torch.jit.script(f2)
try:
    script_f2(inp1, inp2)
except:
    tb.print_exc()

######################################################################
# 하지만 ``torch.compile``은 ``f2``를 쉽게 처리할 수 있습니다.

compile_f2 = torch.compile(f2)
print("compile 2:", test_fns(f2, compile_f2, (inp1, inp2)))
print("~" * 10)

######################################################################
# 이전 컴파일러 솔루션과 비교했을 때 ``torch.compile``이 잘 처리하는
# 또 다른 경우는 PyTorch가 아닌 함수의 사용입니다.

import scipy
def f3(x):
    x = x * 2
    x = scipy.fft.dct(x.numpy())
    x = torch.from_numpy(x)
    x = x * 2
    return x

######################################################################
# TorchScript 추적은 PyTorch가 아닌 함수 호출의 결과를 상수로 취급하므로
# 경고 없이 잘못된 결과가 나올 수 있습니다.

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)
traced_f3 = torch.jit.trace(f3, (inp1,))
print("traced 3:", test_fns(f3, traced_f3, (inp2,)))

######################################################################
# TorchScript scripting과 FX 추적은 PyTorch가 아닌 함수 호출을 허용하지 않습니다.

try:
    torch.jit.script(f3)
except:
    tb.print_exc()

try:
    torch.fx.symbolic_trace(f3)
except:
    tb.print_exc()

######################################################################
# 이에 비해 ``torch.compile``은 PyTorch가 아닌 함수 호출을 쉽게 처리할 수 있습니다.

compile_f3 = torch.compile(f3)
print("compile 3:", test_fns(f3, compile_f3, (inp2,)))

######################################################################
# TorchDynamo와 FX 그래프
# --------------------------
#
# ``torch.compile``의 중요한 구성 요소 중 하나는 TorchDynamo입니다.
# TorchDynamo는 임의의 Python 코드를 JIT 컴파일하여
# `FX 그래프 <https://pytorch.org/docs/stable/fx.html#torch.fx.Graph>`__\ 로 만드는 역할을 하며
# 이후 이 그래프를 더 최적화할 수 있습니다. TorchDynamo는 런타임 중에 Python 바이트코드를
# 분석하고 PyTorch 연산 호출을 감지하여 FX 그래프를 추출합니다.
#
# 일반적으로 ``torch.compile``의 또 다른 구성 요소인 TorchInductor는
# FX 그래프를 최적화된 커널로 추가 컴파일하지만
# TorchDynamo는 다양한 백엔드를 사용할 수 있도록 합니다. TorchDynamo가 출력하는
# FX 그래프를 살펴보기 위해 FX 그래프를 출력하고 그래프의 최적화되지 않은 forward 메소드를
# 그대로 반환하는 사용자 정의 백엔드를 만들어보겠습니다.

from typing import List
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# 다른 백엔드를 사용하므로 초기화합니다.
torch._dynamo.reset()

opt_model = torch.compile(init_model(), backend=custom_backend)
opt_model(generate_data(16)[0])

######################################################################
# 이제 사용자 정의 백엔드를 사용하여 TorchDynamo가 데이터 의존 제어 흐름을
# 어떻게 처리할 수 있는지 확인할 수 있습니다. 아래 함수를 살펴보면
# ``if b.sum() < 0`` 줄이 데이터 의존 제어 흐름의 원인입니다.

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

opt_bar = torch.compile(bar, backend=custom_backend)
inp1 = torch.randn(10)
inp2 = torch.randn(10)
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)

######################################################################
# 출력 결과를 보면 TorchDynamo가 다음 코드에 해당하는 서로 다른 FX 그래프 3개를
# 추출했음을 알 수 있습니다(순서는 위 출력과 다를 수 있습니다).
#
# 1. ``x = a / (torch.abs(a) + 1)``
# 2. ``b = b * -1; return x * b``
# 3. ``return x * b``
#
# TorchDynamo가 데이터 의존 제어 흐름과 같이 지원하지 않는 Python 기능을 만나면
# 연산 그래프를 분리하고 지원하지 않는 코드는 기본 Python 인터프리터가 처리하도록 한 뒤
# 그래프 캡처를 다시 시작합니다.
#
# TorchDynamo가 ``bar``를 어떻게 단계별로 실행하는지 예제로 살펴보겠습니다.
# ``b.sum() < 0``이면 TorchDynamo는 그래프 1을 실행하고 Python이 조건문의
# 결과를 결정하도록 한 뒤 그래프 2를 실행합니다. 반면 ``not b.sum() < 0``이면
# TorchDynamo는 그래프 1을 실행하고 Python이 조건문의 결과를 결정하도록 한 뒤
# 그래프 3을 실행합니다.
#
# 이는 TorchDynamo와 이전 PyTorch 컴파일러 솔루션 사이의 주요 차이점을 보여줍니다.
# 지원하지 않는 Python 기능을 만나면 이전 솔루션은 오류를 발생시키거나 조용히 실패합니다.
# 반면 TorchDynamo는 연산 그래프를 분리합니다.
#
# ``torch._dynamo.explain``을 사용하면 TorchDynamo가 그래프를 어디서 분리하는지 확인할 수 있습니다.

# 다른 백엔드를 사용하므로 초기화합니다.
torch._dynamo.reset()
explain_output = torch._dynamo.explain(bar)(torch.randn(10), torch.randn(10))
print(explain_output)

######################################################################
# 속도 향상을 극대화하려면 그래프 분리를 제한해야 합니다.
# ``fullgraph=True``를 사용하면 TorchDynamo가 처음 만나는 그래프 분리에서
# 오류를 발생시키도록 강제할 수 있습니다.

opt_bar = torch.compile(bar, fullgraph=True)
try:
    opt_bar(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()

######################################################################
# 아래에서는 TorchDynamo가 앞에서 속도 향상을 보여주기 위해 사용한
# 모델에서 그래프를 분리하지 않는다는 것을 보여줍니다.

opt_model = torch.compile(init_model(), fullgraph=True)
print(opt_model(generate_data(16)[0]))

######################################################################
# ``torch.export`` (PyTorch 2.1 이상)를 사용하면 입력 PyTorch 프로그램에서
# 내보낼 수 있는 단일 FX 그래프를 추출할 수 있습니다. 내보낸 그래프는
# 서로 다른 환경, 즉 Python이 없는 환경에서 실행하는 것을 목적으로 합니다.
# 한 가지 중요한 제약은 ``torch.export``가 그래프 분리를 지원하지 않는다는 점입니다.
# ``torch.export``에 대한 자세한 내용은
# `이 튜토리얼 <https://tutorials.pytorch.kr/intermediate/torch_export_tutorial.html>`__\ 을 참고하세요.

######################################################################
# 마무리
# ------------
#
# 이 튜토리얼에서는 기본 사용법을 다루고 eager 모드와 비교한 속도 향상을 보여주며
# 이전 PyTorch 컴파일러 솔루션과 비교하고 TorchDynamo와 FX 그래프의 상호작용을
# 간단히 살펴보면서 ``torch.compile``을 소개했습니다. ``torch.compile``을 한번 사용해 보세요!
