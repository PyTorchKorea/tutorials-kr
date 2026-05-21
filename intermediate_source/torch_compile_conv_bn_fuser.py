# -*- coding: utf-8 -*-
"""
torch.compile 기반 합성곱·배치 정규화 퓨저(Convolution/Batch Norm fuser) 만들기
===========================================================

**저자:** `Horace He <https://github.com/chillee>`_, `Will Feng <https://github.com/yf225>`_ 번역: `심기택 <https://github.com/skt0725>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` 배울 내용
       :class-card: card-prerequisites

       * torch.compile의 패턴 매처에 커스텀 퓨전 패턴을 등록하는 방법

    .. grid-item-card:: :octicon:`list-unordered;1em;` 전제 조건
       :class-card: card-prerequisites

       * PyTorch v2.7.0

.. note::
   이 최적화는 추론 모드의 모델에만 적용됩니다 (예: ``model.eval()``).
   하지만 torch.compile의 패턴 매칭 시스템은 학습과 추론 모두에서 동작합니다.

"""


######################################################################
# 먼저 이후 코드에서 사용할 모듈들을 import 하겠습니다.

from typing import Type, Dict, Any, Tuple, Iterable
import copy
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# 이번 튜토리얼에서는 합성곱과 배치 정규화로 구성된 모델을 만들어 보겠습니다.
# 이 모델에는 몇 가지 까다로운 요소가 있다는 점에 유의하세요.
# 일부 합성곱·배치 정규화 패턴은 Sequential 내부에 숨겨져 있으며, 배치 정규화 중 하나는 또 다른
# Module로 감싸져 있습니다.

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

model = M().to(device)
model.eval()

######################################################################
# 합성곱과 배치 정규화 퓨전하기
# -----------------------------------------
# 합성곱과 배치 정규화를 자동으로 퓨전하려 할 때의 주요 어려움 중 하나는 PyTorch가 계산
# 그래프(computational graph)에 쉽게 접근할 수 있는 방법을 제공하지 않는다는 점입니다.
# torch.compile은 컴파일 과정에서 계산 그래프를 확보함으로써 이 문제를 해결하며,
# 이를 통해 Sequential 모듈 내부에 있는 중첩된 연산이나 사용자 정의 모듈로 감싸진 연산을 포함한
# 모델 전체에 패턴 기반 최적화를 적용할 수 있습니다.
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import register_replacement

######################################################################
# torch.compile은 모델의 계산 그래프를 확보합니다.
# 컴파일 과정에서 Sequential 컨테이너에 숨겨진 모듈과 다른 모듈로 감싸진 모듈들은 모두 그래프에
# 직접 포함되어 패턴 매칭과 최적화의 대상이 됩니다.


######################################################################
# 합성곱과 배치 정규화 퓨전하기
# ----------------------------------
# 다른 일부 퓨전과 달리, 합성곱과 배치 정규화의 퓨전에는 새로운 연산자가 필요하지 않습니다.
# 추론 과정에서 배치 정규화는 요소별 덧셈과 곱셈으로 이루어지므로 이러한 연산들을 앞선 합성곱의 가중치에
# 반영할 수 있습니다. 이를 통해 모델에서 배치 정규화를 완전히 제거할 수 있습니다!
# 자세한 내용은 이 글을 참고하세요.
# https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
# 여기서 사용한 코드는 설명의 명확성을 위해 다음의 구현을 가져온 것입니다. https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/nn/utils/fusion.py
def fuse_conv_bn_eval(conv, bn):
    """
    합성곱 모듈 A와 배치 정규화 모듈 B가 주어졌을 때, 추론 모드에서 C(x) == B(A(x))를 만족하는
    합성곱 모듈 C를 반환합니다.
    """
    assert(not (conv.training or bn.training)), "추론 모드에서만 퓨전합니다!"
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


######################################################################
# torch.compile 기반 패턴 매칭
# ------------------------------------
# 이제 퓨전 로직을 구현했으므로 컴파일 과정에서 torch.compile의 패턴 매처가 인식하고 치환할 수 있는
# 패턴을 등록해야 합니다.

# 매칭하려는 패턴을 정의합니다: conv2d 다음에 batch_norm이 오는 패턴입니다.
def conv_bn_pattern(x, conv_weight, conv_bias, bn_mean, bn_var, bn_weight, bn_bias):
    conv_out = torch.nn.functional.conv2d(x, conv_weight, conv_bias)
    bn_out = torch.nn.functional.batch_norm(
        conv_out, bn_mean, bn_var, bn_weight, bn_bias,
        training=False, eps=1e-5
    )
    return bn_out

def conv_bn_replacement(x, conv_weight, conv_bias, bn_mean, bn_var, bn_weight, bn_bias):
    fused_weight, fused_bias = fuse_conv_bn_weights(
        conv_weight, conv_bias, bn_mean, bn_var, 1e-5, bn_weight, bn_bias
    )
    return torch.nn.functional.conv2d(x, fused_weight, fused_bias)

# 패턴 함수들을 추적하려면 예시 입력이 필요합니다.
# 이 입력들은 conv_bn_pattern 및 conv_bn_replacement의 함수 시그니처와 일치해야 합니다.
# 이들은 패턴 함수를 추적하여 매치 템플릿을 만드는 데 사용됩니다.
# 중요: 패턴 매처는 입력 형태에 구애받지 않습니다! 여기서 사용하는 특정 형태가 매칭될 형태를 제한하지 않습니다.
# 채널, 커널 크기, 공간 차원에 관계없이 유효한 conv2d -> batch_norm 시퀀스라면 모두 매칭됩니다.
# - x: 입력 텐서 (배치 크기, 채널, 높이, 너비)
# - conv_weight: (출력 채널, 입력 채널, 커널 높이, 커널 너비)
# - conv_bias: (출력 채널,)
# - bn_mean, bn_var, bn_weight, bn_bias: 모두 출력 채널과 일치하는 형태(num_features,)를 가집니다.
example_inputs = [
    torch.randn(1, 1, 4, 4).to(device),  # x: 입력 텐서
    torch.randn(1, 1, 1, 1).to(device),  # conv_weight: 출력 채널 1, 입력 채널 1, 1x1 커널
    torch.randn(1).to(device),           # conv_bias: 출력 채널 1
    torch.randn(1).to(device),           # bn_mean: 배치 정규화 이동 평균
    torch.randn(1).to(device),           # bn_var: 배치 정규화 이동 분산
    torch.randn(1).to(device),           # bn_weight: 배치 정규화 가중치 (감마)
    torch.randn(1).to(device),           # bn_bias: 배치 정규화 편향 (베타)
]

from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._inductor import config

# 패턴 매처 패스를 생성하고 패턴을 등록합니다.
patterns = PatternMatcherPass()

register_replacement(
    conv_bn_pattern,
    conv_bn_replacement,
    example_inputs,
    pm.fwd_only,
    patterns,
)

# 등록된 패턴을 적용하는 커스텀 패스 함수를 생성합니다.
def conv_bn_fusion_pass(graph):
    return patterns.apply(graph)

# 설정에 커스텀 패스를 지정합니다.
config.post_grad_custom_post_pass = conv_bn_fusion_pass


######################################################################
# .. 참고::
#       설명을 돕기 위해 2D 합성곱 연산만 매칭하는 등 일부 단순화를 적용하였습니다.
#       torch.compile의 패턴 매처는 이보다 훨씬 더 복잡한 패턴도 처리할 수 있습니다.

######################################################################
# 퓨전 패스 테스트하기
# -----------------------------------------
# 앞서 만든 토이 모델에 이 퓨전 패스를 실행하여 결과가 기존과 완벽히 동일한지 확인할 수 있습니다.
# 또한, 퓨전이 완료된 모델의 코드를 직접 출력해 봄으로써 배치 정규화 연산이 정말로 모두 제거되었는지
# 검증할 수 있습니다.

from torch._dynamo.utils import counters

# 컴파일하기 전에 카운터를 초기화합니다.
counters.clear()

# 패턴 매처가 활성화되어 있는지 확인합니다.
config.pattern_matcher = True

fused_model = torch.compile(model, backend="inductor")
inp = torch.randn(5, 1, 1, 1).to(device)

# 모델을 실행하여 컴파일과 패턴 매칭 과정을 동작시킵니다.
with torch.no_grad():
    output = fused_model(inp)
    expected = model(inp)
    torch.testing.assert_close(output, expected)

# 몇 개의 패턴이 매칭되었는지 확인합니다.
assert counters['inductor']['pattern_matcher_count'] == 3, "3개의 conv-bn 패턴이 매칭될 것으로 예상됩니다."

# 앞선 예시 입력과는 다른 형태를 가진 모델을 만듭니다.
test_model_diff_shape = nn.Sequential(
    nn.Conv2d(3, 16, 5),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 32, 7),
    nn.BatchNorm2d(32),
).to(device).eval()

counters.clear()
compiled_diff_shape = torch.compile(test_model_diff_shape, backend="inductor")
test_input_diff_shape = torch.randn(1, 3, 28, 28).to(device)
with torch.no_grad():
    compiled_diff_shape(test_input_diff_shape)

# 몇 개의 패턴이 매칭되었는지 확인합니다.
assert counters['inductor']['pattern_matcher_count'] == 2, "2개의 conv-bn 패턴이 매칭될 것으로 예상됩니다."


######################################################################
# ResNet18 모델을 사용한 퓨전 성능 측정
# -----------------------------------
# ResNet18과 같은 더 큰 모델에 퓨전 패스를 테스트하여
# 이 단계가 추론 성능을 얼마나 향상시키는지 확인할 수 있습니다.
import torchvision.models as models
import time

rn18 = models.resnet18().to(device)
rn18.eval()

inp = torch.randn(10, 3, 224, 224).to(device)
output = rn18(inp)

def benchmark(model, iters=20):
    with torch.no_grad():
        for _ in range(10):
            model(inp)
        begin = time.time()
        for _ in range(iters):
            model(inp)
        return str(time.time()-begin)

# 원본 모델의 성능을 측정합니다.
print("Original model time: ", benchmark(rn18))

# 앞서 정의한 커스텀 패턴을 적용하여 컴파일합니다.
compiled_with_pattern_matching = torch.compile(rn18, backend="inductor")

# 컴파일된 모델의 성능을 측정합니다.
print("\ntorch.compile (with conv-bn pattern matching and other fusions): ", benchmark(compiled_with_pattern_matching))


######################################################################
# 결론
# ----------
# 보시다시피 torch.compile은 패턴 매칭을 통해 그래프 변환 및 최적화를 구현하는 매우 강력한 방법을
# 제공합니다. 커스텀 패턴을 등록함으로써 torch.compile의 최적화 기능을 더욱 확장하여 특정 도메인에
# 특화된 변환까지 처리할 수 있습니다.
#
# 여기서 보여드린 conv-bn 퓨전은 torch.compile의 패턴 매칭 시스템으로 할 수 있는
# 많은 일들 중 하나의 예시일 뿐입니다.
