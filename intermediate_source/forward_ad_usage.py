# -*- coding: utf-8 -*-
"""
순전파 모드 자동 미분(Beta)
=============================================

**번역**: `김경민 <https://github.com/BcKmini>`_

이 튜토리얼은 순전파 모드 자동 미분(Forward-mode Automatic Differentiation)을 사용하여 방향성 도함수(directional derivative) 또는 야코비안-벡터 곱(Jacobian-vector product)을 계산하는 방법을 보여줍니다.

아래 튜토리얼은 1.11 이상 버전(또는 나이틀리 빌드)에서만 사용할 수 있는 일부 API를 사용합니다.

또한, 순전파 모드 자동 미분은 현재 베타 버전입니다. 따라서 API가 변경될 수 있으며, 아직 일부 연산자는 지원되지 않을 수 있습니다.

기본 사용법
--------------------------------------------------------------------
역전파 모드 자동 미분(Reverse-mode Automatic Differentiation)과 달리, 순전파 모드 자동 미분은 순전파(forward pass)를 진행하며 기울기(gradient)를 즉시(계산을 미루지 않고) 계산합니다. 순전파 모드 자동 미분으로 방향성 도함수를 계산하려면, 먼저 입력을 방향성 도함수의 방향을 나타내는 다른 tensor(야코비안-벡터 곱의 `v`에 해당)와 연결한 뒤 이전과 같이 순전파를 수행하면 됩니다. 'primal'이라고 부르는 입력이 'tangent'라고 부르는 '방향' tensor와 연결될 때, 결과로 나오는 새로운 tensor 객체는 이중수(dual numbers) [0] 와의 관련성 때문에 '이중 tensor(dual tensor)'라고 불립니다.

순전파가 수행될 때, 입력 tensor 중 하나라도 이중 tensor이면 함수의 '민감도(sensitivity)'를 전파하기 위해 추가적인 연산이 수행됩니다.

"""

import torch
import torch.autograd.forward_ad as fwAD

primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# 모든 순전파 자동 미분 연산은 ``dual_level`` 컨텍스트 안에서 수행해야 합니다.
# 이 컨텍스트에서 생성된 모든 이중 tensor의 탄젠트는 컨텍스트를 벗어날 때 소멸됩니다.
# 이는 해당 연산의 출력이나 중간 결과가 향후 다른 순전파 자동 미분 연산에 재사용될 때,
# 현재 연산에 속한 탄젠트가 나중 연산의 탄젠트와 혼동되는 것을 방지하기 위함입니다.
with fwAD.dual_level():
    # 이중 tensor를 만들려면 'primal' tensor를 같은 크기의 다른 tensor,
    # 즉 '탄젠트(tangent)'와 연결합니다.
    # 만약 탄젠트의 레이아웃이 primal의 레이아웃과 다르면,
    # 탄젠트의 값은 primal과 동일한 메타데이터를 갖는 새 tensor에 복사됩니다.
    # 그렇지 않으면 탄젠트 자체가 그대로 사용됩니다.
    #
    # ``make_dual`` 로 생성된 이중 tensor는 primal tensor의 **뷰(데이터를 공유하는 참조)** 라는 점도
    # 중요합니다.
    dual_input = fwAD.make_dual(primal, tangent)
    assert fwAD.unpack_dual(dual_input).tangent is tangent

    # 탄젠트가 복사되는 경우를 보여주기 위해,
    # primal과 다른 레이아웃을 가진 탄젠트를 전달합니다.
    dual_input_alt = fwAD.make_dual(primal, tangent.T)
    assert fwAD.unpack_dual(dual_input_alt).tangent is not tangent

    # 탄젠트가 연결되지 않은 tensor는 자동으로
    # 같은 shape을 가지며 0으로 채워진 탄젠트를 가진 것으로 간주됩니다.
    plain_tensor = torch.randn(10, 10)
    dual_output = fn(dual_input, plain_tensor)

    # 이중 tensor를 풀면(unpack) ``primal`` 과 ``tangent`` 를
    # 속성으로 갖는 ``namedtuple`` 이 반환됩니다.
    jvp = fwAD.unpack_dual(dual_output).tangent

assert fwAD.unpack_dual(dual_output).tangent is None

######################################################################
# 모듈과 함께 사용하기
# --------------------------------------------------------------------
# ``nn.Module`` 을 순전파 자동 미분과 함께 사용하려면, 순전파를 수행하기 전에
# 모델의 매개변수(parameter)를 이중 tensor로 교체해야 합니다. 현재 이중 tensor로 된
# `nn.Parameter` 는 생성할 수 없습니다. 이에 대한 해결 방법으로,
# 이중 tensor를 모듈의 매개변수가 아닌 일반 속성으로 등록해야 합니다.

import torch.nn as nn

model = nn.Linear(5, 5)
input = torch.randn(16, 5)

params = {name: p for name, p in model.named_parameters()}
tangents = {name: torch.rand_like(p) for name, p in params.items()}

with fwAD.dual_level():
    for name, p in params.items():
        delattr(model, name)
        setattr(model, name, fwAD.make_dual(p, tangents[name]))

    out = model(input)
    jvp = fwAD.unpack_dual(out).tangent

######################################################################
# 함수형 모듈 API 사용하기 (Beta)
# --------------------------------------------------------------------
# ``nn.Module`` 을 순전파 자동 미분과 함께 사용하는 또 다른 방법은
# 함수형 모듈 API를 활용하는 것입니다. (상태가 없는 모듈 API라고도 함)

from torch.func import functional_call

# functional_call은 모델에 매개변수가 등록되어 있어야 하므로
# 새로운 모듈이 필요합니다.
model = nn.Linear(5, 5)

dual_params = {}
with fwAD.dual_level():
    for name, p in params.items():
        # 위 섹션과 동일한 ``tangents`` 를 사용합니다.
        dual_params[name] = fwAD.make_dual(p, tangents[name])
    out = functional_call(model, dual_params, input)
    jvp2 = fwAD.unpack_dual(out).tangent

# 결과 확인
assert torch.allclose(jvp, jvp2)

######################################################################
# 사용자 정의 autograd 함수
# --------------------------------------------------------------------
# 사용자 정의 함수 또한 순전파 모드 자동 미분을 지원합니다. 순전파 모드 자동 미분을
# 지원하는 사용자 정의 함수를 만들려면 ``jvp()`` 정적 메소드를
# 등록해야 합니다. 사용자 정의 함수가 순전파와 역전파 자동 미분을 모두 지원하는 것도
# 가능하지만 필수는 아닙니다. 더 자세한 정보는
# `문서 <https://pytorch.org/docs/master/notes/extending.html#forward-mode-ad>`_
# 를 참고하세요.

class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        result = torch.exp(foo)
        # ``ctx`` 에 저장된 tensor는 이후의 순전파 기울기
        # 계산에 사용할 수 있습니다.
        ctx.result = result
        return result

    @staticmethod
    def jvp(ctx, gI):
        gO = gI * ctx.result
        # ``ctx`` 에 저장된 tensor가 역전파에 사용되지 않을 경우,
        # ``del`` 을 사용하여 수동으로 메모리에서 해제할 수 있습니다.
        del ctx.result
        return gO

fn = Fn.apply

primal = torch.randn(10, 10, dtype=torch.double, requires_grad=True)
tangent = torch.randn(10, 10)

with fwAD.dual_level():
    dual_input = fwAD.make_dual(primal, tangent)
    dual_output = fn(dual_input)
    jvp = fwAD.unpack_dual(dual_output).tangent

# 사용자 정의 autograd 함수가 기울기를 올바르게 계산하는지 확인하려면
# ``autograd.gradcheck`` 를 사용하는 것이 중요합니다. 기본적으로
# ``gradcheck`` 는 역전파 모드(reverse-mode) 자동 미분 기울기만 확인합니다.
# ``check_forward_ad=True`` 를 지정하여 순전파 기울기도 확인하도록 할 수 있습니다.
# 만약 함수에 대한 역전파를 구현하지 않았다면, ``check_backward_ad=False``,
# ``check_undefined_grad=False``, ``check_batched_grad=False`` 를 지정하여
# ``gradcheck`` 가 역전파 모드 자동 미분이 필요한 테스트를 건너뛰도록 할 수 있습니다.
torch.autograd.gradcheck(Fn.apply, (primal,), check_forward_ad=True,
                         check_backward_ad=False, check_undefined_grad=False,
                         check_batched_grad=False)

######################################################################
# 함수형 API (Beta)
# --------------------------------------------------------------------
# Functorch는 야코비안-벡터 곱을 계산하기 위한 고수준 함수형 API도
# 제공하며, 사용 사례에 따라 더 간단하게 사용할 수 있습니다.
#
# 함수형 API의 장점은 저수준의 이중 tensor API를 이해하거나 사용할
# 필요가 없으며, 다른 `functorch 변환(vmap 등)과 결합 <https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html>`_
# 할 수 있다는 것입니다. 단점은 세밀한 제어가 어렵다는 점입니다.
#
# 이 튜토리얼의 나머지 부분을 실행하려면 functorch
# (https://github.com/pytorch/functorch) 가 필요합니다.
# 설치 방법은 해당 링크에서 확인해주세요.

import functorch as ft

primal0 = torch.randn(10, 10)
tangent0 = torch.randn(10, 10)
primal1 = torch.randn(10, 10)
tangent1 = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# 위 함수의 JVP를 계산하는 기본 예제입니다.
# ``jvp(func, primals, tangents)`` 는 ``func(*primals)`` 의 결과와 계산된
# 야코비안-벡터 곱(JVP)을 함께 반환합니다. 각 primal은 같은 shape의 탄젠트와
# 연결되어야 합니다.
primal_out, tangent_out = ft.jvp(fn, (primal0, primal1), (tangent0, tangent1))

# ``functorch.jvp`` 는 모든 primal이 탄젠트와 연결될 것을 요구합니다.
# 만약 ``fn`` 의 특정 입력에만 탄젠트를 연결하고 싶다면,
# 탄젠트가 없는 입력을 받는 새로운 함수를 만들어야 합니다.
primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)
y = torch.randn(10, 10)

import functools
new_fn = functools.partial(fn, y=y)
primal_out, tangent_out = ft.jvp(new_fn, (primal,), (tangent,))

######################################################################
# 함수형 API를 모듈과 함께 사용하기
# --------------------------------------------------------------------
# ``nn.Module`` 과 ``functorch.jvp`` 를 함께 사용하여 모델 매개변수에 대한
# 야코비안-벡터 곱을 계산하려면, ``nn.Module`` 을 모델 매개변수와
# 모듈의 입력을 모두 인자로 받는 함수로 재구성해야 합니다.

model = nn.Linear(5, 5)
input = torch.randn(16, 5)
tangents = tuple([torch.rand_like(p) for p in model.parameters()])

# ``ft.make_functional_with_buffers`` 는 주어진 ``torch.nn.Module`` 에서
# 상태(``params`` 와 버퍼)를 추출하고, 함수처럼 호출할 수 있는
# 함수형 버전의 모델을 반환합니다.
# 즉, 반환된 ``func`` 는 ``func(params, buffers, input)`` 처럼 호출할 수 있습니다.
# ``ft.make_functional_with_buffers`` 는 이전에 보았던 ``nn.Module`` 의 상태 없는 API와
# 유사하며, 이 둘을 통합하는 작업이 진행 중입니다.
func, params, buffers = ft.make_functional_with_buffers(model)

# ``jvp`` 는 모든 입력이 탄젠트와 연결될 것을 요구하므로,
# 매개변수를 받았을 때 출력을 생성하는 새로운 함수를 만들어야 합니다.
def func_params_only(params):
    return func(params, buffers, input)

model_output, jvp_out = ft.jvp(func_params_only, (params,), (tangents,))

######################################################################
# [0] https://en.wikipedia.org/wiki/Dual_number