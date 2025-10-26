# -*- coding: utf-8 -*-
"""
순전파 모드 자동 미분 (Beta)
=============================================
**번역**: `김경민 <https://github.com/BcKmini>`_

이 튜토리얼은 순전파 모드 자동 미분(Forward-mode Automatic Differentiation)을 사용하여
방향성 도함수(directional derivative) 또는 야코비안-벡터 곱(Jacobian-vector product)을 계산하는 방법을 보여줍니다.

아래 튜토리얼은 PyTorch 1.11 이상 버전(또는 나이틀리 빌드)에서만 사용할 수 있는 일부 API를 사용합니다.

또한, 순전파 모드 자동 미분은 현재 베타 버전입니다. 따라서 API가 변경될 수 있으며, 아직 일부 연산자는 지원되지 않을 수 있습니다.

기본 사용법
--------------------------------------------------------------------
역전파 모드 자동 미분(Reverse-mode Automatic Differentiation)과 달리,
순전파 모드 자동 미분은 순전파를 진행하며 기울기를 즉시 계산합니다.

순전파 모드 자동 미분으로 방향성 도함수를 계산하려면,
먼저 입력을 방향성 도함수의 방향을 나타내는 다른 텐서(야코비안-벡터 곱의 `v`에 해당)와 연결한 뒤
이전과 같이 순전파를 수행하면 됩니다.

입력 텐서(이하 primal)가 방향 텐서(tangent)와 연결될 때,
결과로 생성되는 텐서 객체를 **이중 텐서(dual tensor)** 라고 부릅니다.
이는 이중수(dual number)[0] 개념과 관련이 있습니다.

순전파가 수행될 때, 입력 텐서 중 하나라도 이중 텐서이면
함수의 민감도(sensitivity)를 전파하기 위해 추가적인 연산이 수행됩니다.
"""

import torch
import torch.autograd.forward_ad as fwAD

primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# 모든 순전파 자동 미분 연산은 ``dual_level`` 컨텍스트 안에서 수행해야 합니다.
# 이 컨텍스트에서 생성된 모든 이중 텐서의 tangent는 컨텍스트를 벗어날 때 소멸됩니다.
# 이는 해당 연산의 출력이나 중간 결과가 향후 다른 순전파 자동 미분 연산에 재사용될 때,
# 현재 연산에 속한 tangent가 나중 연산의 tangent와 혼동되는 것을 방지하기 위함입니다.
with fwAD.dual_level():
    # 이중 텐서를 만들려면 primal 텐서를 같은 크기의 tangent 텐서와 연결합니다.
    dual_input = fwAD.make_dual(primal, tangent)
    assert fwAD.unpack_dual(dual_input).tangent is tangent

    # tangent의 레이아웃이 primal과 다르면,
    # tangent의 값은 primal과 동일한 메타데이터를 갖는 새 텐서에 복사됩니다.
    dual_input_alt = fwAD.make_dual(primal, tangent.T)
    assert fwAD.unpack_dual(dual_input_alt).tangent is not tangent

    # tangent가 연결되지 않은 텐서는 자동으로
    # 같은 shape을 가지며 0으로 채워진 tangent를 가진 것으로 간주됩니다.
    plain_tensor = torch.randn(10, 10)
    dual_output = fn(dual_input, plain_tensor)

    # 이중 텐서를 풀면(unpack) ``primal`` 과 ``tangent`` 를
    # 속성으로 갖는 ``namedtuple`` 이 반환됩니다.
    jvp = fwAD.unpack_dual(dual_output).tangent

assert fwAD.unpack_dual(dual_output).tangent is None


######################################################################
# 모듈과 함께 사용하기
# --------------------------------------------------------------------
# ``nn.Module`` 을 순전파 자동 미분과 함께 사용하려면,
# 순전파를 수행하기 전에 모델의 매개변수(parameter)를 이중 텐서로 교체해야 합니다.
# 현재 이중 텐서로 된 ``nn.Parameter`` 는 직접 생성할 수 없습니다.
# 따라서 이중 텐서를 모듈의 매개변수가 아닌 일반 속성으로 등록해야 합니다.
######################################################################

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
# 함수형 모듈 API를 활용하는 것입니다.
######################################################################

from torch.func import functional_call

model = nn.Linear(5, 5)

dual_params = {}
with fwAD.dual_level():
    for name, p in params.items():
        dual_params[name] = fwAD.make_dual(p, tangents[name])
    out = functional_call(model, dual_params, input)
    jvp2 = fwAD.unpack_dual(out).tangent

assert torch.allclose(jvp, jvp2)


######################################################################
# 사용자 정의 자동 미분 함수
# --------------------------------------------------------------------
# 사용자 정의 함수 또한 순전파 모드 자동 미분을 지원할 수 있습니다.
# 순전파 모드 자동 미분을 지원하는 사용자 정의 함수를 만들려면
# ``jvp()`` 정적 메서드를 등록해야 합니다.
######################################################################

class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        result = torch.exp(foo)
        ctx.result = result
        return result

    @staticmethod
    def jvp(ctx, gI):
        gO = gI * ctx.result
        del ctx.result
        return gO

fn = Fn.apply

primal = torch.randn(10, 10, dtype=torch.double, requires_grad=True)
tangent = torch.randn(10, 10)

with fwAD.dual_level():
    dual_input = fwAD.make_dual(primal, tangent)
    dual_output = fn(dual_input)
    jvp = fwAD.unpack_dual(dual_output).tangent

torch.autograd.gradcheck(Fn.apply, (primal,), check_forward_ad=True,
                         check_backward_ad=False, check_undefined_grad=False,
                         check_batched_grad=False)


######################################################################
# 함수형 API (Beta)
# --------------------------------------------------------------------
# Functorch는 야코비안-벡터 곱을 계산하기 위한 고수준 함수형 API를 제공합니다.
# 저수준 이중 텐서 API를 직접 다루지 않아도 되며, vmap 등 다른 변환과도 결합할 수 있습니다.
######################################################################

import functorch as ft

primal0 = torch.randn(10, 10)
tangent0 = torch.randn(10, 10)
primal1 = torch.randn(10, 10)
tangent1 = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

primal_out, tangent_out = ft.jvp(fn, (primal0, primal1), (tangent0, tangent1))

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
# 야코비안-벡터 곱을 계산하려면, ``nn.Module`` 을
# 매개변수와 입력을 모두 인자로 받는 함수로 구성해야 합니다.
######################################################################

model = nn.Linear(5, 5)
input = torch.randn(16, 5)
tangents = tuple([torch.rand_like(p) for p in model.parameters()])

func, params, buffers = ft.make_functional_with_buffers(model)

def func_params_only(params):
    return func(params, buffers, input)

model_output, jvp_out = ft.jvp(func_params_only, (params,), (tangents,))


######################################################################
# [0] https://en.wikipedia.org/wiki/Dual_number
######################################################################
