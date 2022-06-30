# -*- coding: utf-8 -*-
"""
(베타) FX에서 합성곱/배치 정규화(Convolution/Batch Norm) 결합기(Fuser) 만들기
****************************************************************************
**저자**: `Horace He <https://github.com/chillee>`_

**번역:** `오찬희 <https://github.com/kozeldark>`_

이 튜토리얼에서는 PyTorch의 구성 가능한 함수의 변환을 위한 툴킷인 FX를 사용하여 다음을 수행하고자 합니다.

1) 데이터 의존성에서 합성곱/배치 정규화 패턴을 찾습니다.
2) 1번에서 발견된 패턴의 경우 배치 정규화 통계를 합성곱 가중치로 결합합니다(folding).

이 최적화는 추론 모드(즉, `mode.eval()`)의 모델에만 적용된다는 점에 유의하세요.

다음 링크에 있는 결합기를 만들 것입니다.
https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/fx/experimental/fuser.py

"""

######################################################################
# 몇 가지의 import 과정을 먼저 처리해줍시다(나중에 코드에서 모두 사용할 것입니다).

from typing import Type, Dict, Any, Tuple, Iterable
import copy
import torch.fx as fx
import torch
import torch.nn as nn

######################################################################
# 이 튜토리얼에서는 합성곱과 배치 정규화로 구성된 모델을 만들 것입니다.
# 이 모델에는 아래와 같은 까다로운 요소가 있습니다.
# 합성곱/배치 정규화 패턴 중의 일부는 시퀀스에 숨겨져 있고
# 배치 정규화 중 하나는 다른 모듈로 감싸져 있습니다.

class WrappedBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.BatchNorm2d(1)
    def forward(self, x):
        return self.mod(x)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.nested = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, 1),
        )
        self.wrapped = WrappedBatchNorm()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.nested(x)
        x = self.wrapped(x)
        return x

model = M()

model.eval()

######################################################################
# 합성곱과 배치 정규화 결합하기
# -----------------------------
# PyTorch에서 합성곱과 배치 정규화를 자동으로 결합하려고 할 때 가장 큰 어려움 중 하나는
# PyTorch가 계산 그래프에 쉽게 접근할 수 있는 방법을 제공하지 않는다는 것입니다.
# FX는 호출된 실제 연산을 기호적(symbolically)으로 추적하여 이 문제를 해결하므로 순차적 모듈 내에 중첩되거나
# 사용자 정의 모듈로 감싸진 `forward` 호출을 통해 계산을 추적할 수 있습니다.

traced_model = torch.fx.symbolic_trace(model)
print(traced_model.graph)

######################################################################
# 이렇게 하면 모델을 그래프로 나타낼 수 있습니다.
# 순차적 모듈 및 감싸진 모듈 내에 숨겨진 두 모듈이 모두 그래프에 삽입되어 있습니다.
# 이는 기본 추상화 수준이지만 전달 기록기(pass writer)에서 구성할 수 있습니다.
# 자세한 내용은 다음 링크의 FX 개요에서 확인할 수 있습니다.
# https://pytorch.org/docs/master/fx.html#module-torch.fx


####################################
# 합성곱과 배치 정규화 결합하기
# ---------------------------
# 일부 다른 결합과 달리, 합성곱과 배치 정규화의 결합은 새로운 연산자를 필요로 하지 않습니다.
# 대신, 추론 중 배치 정규화는 점별 덧셈과 곱셈으로 구성되므로,
# 이러한 연산은 이전 합성곱의 가중치로 "미리 계산되어 저장(baked)" 될 수 있습니다.
# 이를 통해 배치 정규화를 모델에서 완전히 제거할 수 있습니다!
# 자세한 내용은 다음 링크에서 확인 할 수 있습니다.
# https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
# 이 코드는 명확성을 위해 다음 링크에서 복사한 것입니다.
# https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/nn/utils/fusion.py

def fuse_conv_bn_eval(conv, bn):
    """
    합성곱 모듈 'A'와 배치 정규화 모듈 'B'가 주어지면
    C(x) == B(A(x))를 만족하는 합성곱 모듈 'C'를 추론 모드로 반환합니다.
    """
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


####################################
# FX 결합 전달(pass)
# --------------
# 이제 합성곱과 배치 정규화를 결합하는 방법뿐만 아니라 계산 그래프도 얻었으므로
# 남은 것은 FX 그래프에 절차를 반복하고 원하는 결합을 적용하는 것입니다.

def _parent_name(target : str) -> Tuple[str, str]:
    """
    정규화 된 이름(qualname)을 부모경로(parent path)와 마지막 요소(last atom)로 나눠줍니다.
    예를 들어, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    # 대부분의 FX 전달의 첫 번째 단계는 `GraphModule` 을 얻기 위해
    # 모델을 기호적으로 추적하는 것입니다.
    # 이것은 원래 모델과 기능적으로 동일한 원래 모델의 표현입니다.
    # 단, 이제는 순전파 단계(forward pass)에 대한 그래프 표현도 가지고 있습니다.
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    # FX 작업을 위한 기본 표현은 `그래프(Graph)` 와 `노드(Node)` 입니다.
    # 각 `GraphModule` 에는 연관된 `그래프` 가 있습니다.
    # 이 `그래프` 는 `GraphModule.code` 를 생성하는 것이기도 합니다.
    # `그래프` 자체는 `노드` 객체의 목록으로 표시됩니다.
    # 따라서 그래프의 모든 작업을 반복하기 위해 `그래프` 에서 각 `노드` 에 대해 반복합니다.
    for node in fx_model.graph.nodes:
        # FX IR 에는 일반적으로 모듈, 함수 또는 메소드에 대한
        # 호출 사이트를 나타내는 여러 유형의 노드가 있습니다.
        # 노드의 유형은 `Node.op` 에 의해 결정됩니다.
        if node.op != 'call_module': # 현재 노드가 모듈을 호출하지 않으면 무시할 수 있습니다.
            continue
        # 호출 사이트의 경우, `Node.target` 은 호출되는 모듈/함수/방법을 나타냅니다.
        # 여기서는 'Node.target' 을 확인하여 배치 정규화 모듈인지 확인한 다음
        # `Node.args[0].target` 을 확인하여 입력 `노드` 가 합성곱인지 확인합니다.
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:  # 합성곱 출력은 다른 노드에서 사용됩니다.
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            # 배치 정규화를 합성곱으로 결합했기 때문에
            # 배치 정규화의 사용을 모두 합성곱으로 교체해야 합니다.
            node.replace_all_uses_with(node.args[0])
            # 배치 정규화 사용을 모두 교체했으므로
            # 안전하게 배치 정규화를 제거할 수 있습니다.
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    # 그래프를 수정한 후에는 생성된 코드를 동기화하기 위해 그래프를 다시 컴파일해야 합니다.
    fx_model.recompile()
    return fx_model


######################################################################
# .. note::
#       여기서는 2D 합성곱만 일치시키는 등 시연 목적으로 약간의 단순화를 하였습니다.
#       더 유용한 전달은 다음 링크를 참조하십시오.
#       https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/fuser.py

######################################################################
# 결합 전달(Fusion pass) 실험하기
# --------------------------------
# 이제 아주 작은 초기 모델에 대해 이 결합 전달을 실행해 결과가 동일한지 확인할 수 있습니다.
# 또한 결합 모델의 코드를 출력하여 더 이상 배치 정규화가 없는지 확인할 수 있습니다.


fused_model = fuse(model)
print(fused_model.code)
inp = torch.randn(5, 1, 1, 1)
torch.testing.assert_allclose(fused_model(inp), model(inp))


######################################################################
# ResNet18에서 결합 벤치마킹하기
# ------------------------------
# 이제 ResNet18과 같은 대형 모델에서 결합 전달을 실험하고
# 이 전달이 추론 성능을 얼마나 향상시키는지 확인할 수 있습니다.
import torchvision.models as models
import time

rn18 = models.resnet18()
rn18.eval()

inp = torch.randn(10, 3, 224, 224)
output = rn18(inp)

def benchmark(model, iters=20):
    for _ in range(10):
        model(inp)
    begin = time.time()
    for _ in range(iters):
        model(inp)
    return str(time.time()-begin)

fused_rn18 = fuse(rn18)
print("Unfused time: ", benchmark(rn18))
print("Fused time: ", benchmark(fused_rn18))
######################################################################
# 앞서 살펴본 바와 같이, FX 변환의 출력은 (Torchscriptable) PyTorch 코드입니다.
# 따라서 `jit.script` 를 통해 쉽게 출력하여 성능을 더 높일 수 있습니다.
# 이러한 방식으로 FX 모델 변환은 Torchscript와 아무런 문제 없이 구성됩니다.

jit_rn18 = torch.jit.script(fused_rn18)
print("jit time: ", benchmark(jit_rn18))


######
# 결론
# ---
# FX를 사용하면 PyTorch 코드에 정적 그래프 변환을 쉽게 작성할 수 있습니다.
#
# FX는 아직 베타 버전이기 때문에 FX 사용에 대한 피드백을 보내주시면 감사하겠습니다.
# PyTorch 포럼 (https://discuss.pytorch.org/)
# 이슈 추적기 (https://github.com/pytorch/pytorch/issues)
# 위 두 링크를 사용하여 피드백을 제공해주시면 됩니다.
