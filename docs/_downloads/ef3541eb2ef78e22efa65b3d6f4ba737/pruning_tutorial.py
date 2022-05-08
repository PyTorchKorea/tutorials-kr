# -*- coding: utf-8 -*-
"""
가지치기 기법(Pruning) 튜토리얼
=====================================
**저자**: `Michela Paganini <https://github.com/mickypaganini>`_
**번역** : `안상준 <https://github.com/Justin-A>`_

최첨단 딥러닝 모델들은 굉장히 많은 수의 파라미터값들로 구성되기 때문에, 쉽게 배포되기 어렵습니다.
이와 반대로, 생물학적 신경망들은 효율적으로 희소하게 연결된 것으로 알려져 있습니다.
모델의 정확도가 손상되지 않는 범위에서 메모리, 배터리, 하드웨어 소비량을 줄이고,
기기에 경량화된 모델을 배치하며, 개인이 이용하고 있는 기기에서 프라이버시가 보장되기 위해서는
모델에 포함된 파라미터 수를 줄여 압축하는 최적의 기법을 파악하는 것이 중요합니다.
연구 측면에서는, 가지치기 기법은 굉장히 많은 수의 파라미터값들로 구성된 모델과
굉장히 적은 수의 파라미터값들로 구성된 모델 간 학습 역학 차이를 조사하는데 주로 이용되기도 하며,
하위 신경망 모델과 파라미터값들의 초기화가 운이 좋게 잘 된 케이스를 바탕으로
("`lottery tickets <https://arxiv.org/abs/1803.03635>`_") 신경망 구조를 찾는 기술들에 대해 반대 의견을 제시하기도 합니다.

이번 튜토리얼에서는, ``torch.nn.utils.prune`` 을 이용하여 여러분이 설계한 딥러닝 모델에 대해 가지치기 기법을 적용해보는 것을 배워보고,
심화적으로 여러분의 맞춤형 가지치기 기법을 구현하는 방법에 대해 배워보도록 하겠습니다.

요구사항
------------
``"torch>=1.4"``

"""
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

######################################################################
# 딥러닝 모델 생성
# -----------------------
# 이번 튜토리얼에서는, 얀 르쿤 교수님의 연구진들이 1998년도에 발표한 ``LeNet
# <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`` 의 모델 구조를 이용합니다.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1개 채널 수의 이미지를 입력값으로 이용하여 6개 채널 수의 출력값을 계산하는 방식
        # Convolution 연산을 진행하는 커널(필터)의 크기는 3x3 을 이용
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Convolution 연산 결과 5x5 크기의 16 채널 수의 이미지
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)


######################################################################
# 모듈 점검
# -----------------
#
# 가지치기 기법이 적용되지 않은 LeNet 모델의 ``conv1`` 층을 점검해봅시다.
# 여기에는 2개의 파라미터값들인 ``가중치`` 값과 ``편향`` 값이 포함될 것이며, 버퍼는 존재하지 않을 것입니다.
#

module = model.conv1
print(list(module.named_parameters()))

######################################################################
print(list(module.named_buffers()))

######################################################################
# 모듈 가지치기 기법 적용 예제
# -----------------------------------
#
# 모듈에 대해 가지치기 기법을 적용하기 위해 (이번 예제에서는, LeNet 모델의 ``conv1`` 층)
# 첫 번째로는, ``torch.nn.utils.prune`` (또는 ``BasePruningMethod`` 의 서브 클래스로 직접
# `구현 <torch-nn-utils-prune>`_ )
# 내 존재하는 가지치기 기법을 선택합니다.
# 그 후, 해당 모듈 내에서 가지치기 기법을 적용하고자 하는 모듈과 파라미터를 지정합니다.
# 마지막으로, 가지치기 기법에 적당한 키워드 인자값을 이용하여 가지치기 매개변수를 지정합니다.
# 이번 예제에서는, ``conv1`` 층의 가중치의 30%값들을 랜덤으로 가지치기 기법을 적용해보겠습니다.
# 모듈은 함수에 대한 첫 번째 인자값으로 전달되며, ``name`` 은 문자열 식별자를 이용하여 해당 모듈 내 매개변수를 구분합니다.
# 그리고, ``amount`` 는 가지치기 기법을 적용하기 위한 대상 가중치값들의 백분율 (0과 1사이의 실수값),
# 혹은 가중치값의 연결의 개수 (음수가 아닌 정수) 를 지정합니다.

prune.random_unstructured(module, name="weight", amount=0.3)

######################################################################
# 가지치기 기법은 가중치값들을 파라미터값들로부터 제거하고 ``weight_orig`` (즉, 초기 가중치 이름에 "_orig"을 붙인) 이라는
# 새로운 파라미터값으로 대체하는 것으로 실행됩니다.
# ``weight_orig`` 은 텐서값에 가지치기 기법이 적용되지 않은 상태를 저장합니다.
# ``bias`` 은 가지치기 기법이 적용되지 않았기 때문에 그대로 남아 있습니다.
print(list(module.named_parameters()))

######################################################################
# 위에서 선택한 가지치기 기법에 의해 생성되는 가지치기 마스크는 초기 파라미터  ``name`` 에 ``weight_mask``
# (즉, 초기 가중치 이름에 "_mask"를 붙인) 이름의 모듈 버퍼로 저장됩니다.
print(list(module.named_buffers()))

######################################################################
# 수정이 되지 않은 상태에서 순전파를 진행하기 위해서는 ``가중치`` 값 속성이 존재해야 합니다.
# ``torch.nn.utils.prune`` 내 구현된 가지치기 기법은 가지치기 기법이 적용된 가중치값들을 이용하여
# (기존의 가중치값에 가지치기 기법이 적용된) 순전파를 진행하고, ``weight`` 속성값에 가지치기 기법이 적용된 가중치값들을 저장합니다.
# 이제 가중치값들은 ``module`` 의 매개변수가 아니라 하나의 속성값으로 취급되는 점을 주의하세요.
print(module.weight)

######################################################################
# 최종적으로, 가지치기 기법은 파이토치의 ``forward_pre_hooks`` 를 이용하여 각 순전파가 진행되기 전에 가지치기 기법이 적용됩니다.
# 구체적으로, 지금까지 진행한 것 처럼, 모듈이 가지치기 기법이 적용되었을 때,
# 가지치기 기법이 적용된 각 파라미터값들이 ``forward_pre_hook`` 를 얻게됩니다.
# 이러한 경우, ``weight`` 이름인 기존 파라미터값에 대해서만 가지치기 기법을 적용하였기 때문에,
# 훅은 오직 1개만 존재할 것입니다.
print(module._forward_pre_hooks)

######################################################################
# 완결성을 위해, 편향값에 대해서도 가지치기 기법을 적용할 수 있으며,
# 모듈의 파라미터, 버퍼, 훅, 속성값들이 어떻게 변경되는지 확인할 수 있습니다.
# 또 다른 가지치기 기법을 적용해보기 위해, ``l1_unstructured`` 가지치기 함수에서 구현된 내용과 같이,
# L1 Norm 값이 가장 작은 편향값 3개를 가지치기를 시도해봅시다.
prune.l1_unstructured(module, name="bias", amount=3)

######################################################################
# 이전에서 실습한 내용을 토대로, 명명된 파라미터값들이 ``weight_orig``, ``bias_orig`` 2개를 모두 포함할 것이라 예상할 수 있습니다.
# 버퍼들은 ``weight_mask``, ``bias_mask`` 2개를 포함할 것입니다.
# 가지치기 기법이 적용된 2개의 텐서값들은 모듈의 속성값으로 존재할 것이며, 모듈은 2개의 ``forward_pre_hooks`` 을 갖게 될 것입니다.
print(list(module.named_parameters()))

######################################################################
print(list(module.named_buffers()))

######################################################################
print(module.bias)

######################################################################
print(module._forward_pre_hooks)

######################################################################
# 가지치기 기법 반복 적용
# ------------------------------------
#
# 모듈 내 같은 파라미터값에 대해 가지치기 기법이 여러번 적용될 수 있으며, 다양한 가지치기 기법의 조합이 적용된 것과 동일하게 적용될 수 있습니다.
# 새로운 마스크와 이전의 마스크의 결합은 ``PruningContainer`` 의 ``compute_mask`` 메소드를 통해 처리할 수 있습니다.
#
# 예를 들어, 만약 ``module.weight`` 값에 가지치기 기법을 적용하고 싶을 때, 텐서의 0번째 축의 L2 norm값을 기준으로 구조화된 가지치기 기법을 적용합니다.
# (여기서 0번째 축이란, 합성곱 연산을 통해 계산된 출력값에 대해 각 채널별로 적용된다는 것을 의미합니다.)
# 이 방식은 ``ln_structured`` 함수와 ``n=2`` 와 ``dim=0`` 의 인자값을 바탕으로 구현될 수 있습니다.
prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

############################################################################
# 우리가 확인할 수 있듯이, 이전 마스크의 작용을 유지하면서 채널의 50% (6개 중 3개) 에 해당되는 모든 연결을 0으로 변경합니다.
print(module.weight)

############################################################################
# 이에 해당하는 훅은 ``torch.nn.utils.prune.PruningContainer`` 형태로 존재하며, 가중치에 적용된 가지치기 기법의 이력을 저장합니다.
for hook in module._forward_pre_hooks.values():
    if hook._tensor_name == "weight":  # 가중치에 해당하는 훅을 선택
        break

print(list(hook))  # 컨테이너 내 가지치기 기법의 이력

######################################################################
# 가지치기 기법이 적용된 모델의 직렬화
# ---------------------------------------------
# 마스크 버퍼들과 가지치기 기법이 적용된 텐서 계산에 사용된 기존의 파라미터를 포함하여 관련된 모든 텐서값들은
# 필요한 경우 모델의 ``state_dict`` 에 저장되기 때문에, 쉽게 직렬화하여 저장할 수 있습니다.
print(model.state_dict().keys())


######################################################################
# 가지치기 기법의 재-파라미터화 제거
# -----------------------------------------
#
# 가지치기 기법이 적용된 것을 영구적으로 만들기 위해서, 재-파라미터화 관점의
# ``weight_orig`` 와 ``weight_mask`` 값을 제거하고, ``forward_pre_hook`` 값을 제거합니다.
# 제거하기 위해 ``torch.nn.utils.prune`` 내 ``remove`` 함수를 이용할 수 있습니다.
# 가지치기 기법이 적용되지 않은 것처럼 실행되는 것이 아닌 점을 주의하세요.
# 이는 단지 가지치기 기법이 적용된 상태에서 가중치 파라미터값을 모델 파라미터값으로 재할당하는 것을 통해 영구적으로 만드는 것일 뿐입니다.

######################################################################
# 재-파라미터화를 제거하기 전 상태
print(list(module.named_parameters()))
######################################################################
print(list(module.named_buffers()))
######################################################################
print(module.weight)

######################################################################
# 재-파라미터를 제거한 후 상태
prune.remove(module, 'weight')
print(list(module.named_parameters()))
######################################################################
print(list(module.named_buffers()))

######################################################################
# 모델 내 여러 파라미터값들에 대하여 가지치기 기법 적용
# ----------------------------------------------------------
#
# 가지치기 기법을 적용하고 싶은 파라미터값들을 지정함으로써, 이번 예제에서 볼 수 있는 것 처럼,
# 신경망 모델 내 여러 텐서값들에 대해서 쉽게 가지치기 기법을 적용할 수 있습니다.

new_model = LeNet()
for name, module in new_model.named_modules():
    # 모든 2D-conv 층의 20% 연결에 대해 가지치기 기법을 적용
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # 모든 선형 층의 40% 연결에 대해 가지치기 기법을 적용
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)

print(dict(new_model.named_buffers()).keys())  # 존재하는 모든 마스크들을 확인

######################################################################
# 전역 범위에 대한 가지치기 기법 적용
# ----------------------------------------------
#
# 지금까지, "지역 변수" 에 대해서만 가지치기 기법을 적용하는 방법을 살펴보았습니다.
# (즉, 가중치 규모, 활성화 정도, 경사값 등의 각 항목의 통계량을 바탕으로 모델 내 텐서값 하나씩 가지치기 기법을 적용하는 방식)
# 그러나, 범용적이고 아마 더 강력한 방법은 각 층에서 가장 낮은 20%의 연결을 제거하는것 대신에, 전체 모델에 대해서 가장 낮은 20% 연결을 한번에 제거하는 것입니다.
# 이것은 각 층에 대해서 가지치기 기법을 적용하는 연결의 백분율값을 다르게 만들 가능성이 있습니다.
# ``torch.nn.utils.prune`` 내 ``global_unstructured`` 을 이용하여 어떻게 전역 범위에 대한 가지치기 기법을 적용하는지 살펴봅시다.

model = LeNet()

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

######################################################################
# 이제 각 층에 존재하는 연결들에 가지치기 기법이 적용된 정도가 20%가 아닌 것을 확인할 수 있습니다.
# 그러나, 전체 가지치기 적용 범위는 약 20%가 될 것입니다.
print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    )
)
print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    )
)
print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc2.weight == 0))
        / float(model.fc2.weight.nelement())
    )
)
print(
    "Sparsity in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc3.weight == 0))
        / float(model.fc3.weight.nelement())
    )
)
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)


######################################################################
# ``torch.nn.utils.prune`` 에서 확장된 맞춤형 가지치기 기법
# ------------------------------------------------------------------
# 맞춤형 가지치기 기법은, 다른 가지치기 기법을 적용하는 것과 같은 방식으로,
# ``BasePruningMethod`` 의 기본 클래스인 ``nn.utils.prune`` 모듈을 활용하여 구현할 수 있습니다.
# 기본 클래스는 ``__call__``, ``apply_mask``, ``apply``, ``prune``, ``remove`` 메소드들을 내포하고 있습니다.
# 특별한 케이스가 아닌 경우, 기본적으로 구성된 메소드들을 재구성할 필요가 없습니다.
# 그러나, ``__init__`` (구성요소), ``compute_mask``
# (가지치기 기법의 논리에 따라 주어진 텐서값에 마스크를 적용하는 방법) 을 고려하여 구성해야 합니다.
# 게다가, 가지치기 기법을 어떠한 방식으로 적용하는지 명확하게 구성해야 합니다.
# (지원되는 옵션은 ``global``, ``structured``, ``unstructured`` 입니다.)
# 이러한 방식은, 가지치기 기법을 반복적으로 적용해야 하는 경우 마스크를 결합하는 방법을 결정하기 위해 필요합니다.
# 즉, 이미 가지치기 기법이 적용된 모델에 대해서 가지치기 기법을 적용할 때,
# 기존의 가지치기 기법이 적용되지 않은 파라미터 값에 대해 가지치기 기법이 영향을 미칠 것으로 예상됩니다.
# ``PRUNING_TYPE`` 을 지정한다면, 가지치기 기법을 적용하기 위해 파라미터 값을 올바르게 제거하는
# ``PruningContainer`` (마스크 가지치기 기법을 반복적으로 적용하는 것을 처리하는)를 가능하게 합니다.
# 예를 들어, 다른 모든 항목이 존재하는 텐서를 가지치기 기법을 구현하고 싶을 때,
# (또는, 텐서가 이전에 가지치기 기법에 의해 제거되었거나 남아있는 텐서에 대해)
# 한 층의 개별 연결에 작용하며 전체 유닛/채널 (``'structured'``), 또는 다른 파라미터 간
# (``'global'``) 연결에는 작용하지 않기 때문에 ``PRUNING_TYPE='unstructured'`` 방식으로 진행됩니다.

class FooBarPruningMethod(prune.BasePruningMethod):
    """
    텐서 내 다른 항목들에 대해 가지치기 기법을 적용
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask

######################################################################
# ``nn.Module`` 의 매개변수에 적용하기 위해 인스턴스화하고 적용하는 간단한 기능을 구현해봅니다.
def foobar_unstructured(module, name):
    """
    텐서 내 다른 모든 항목들을 제거하여 `module` 에서 `name` 이라는 파라미터에 대해 가자치기 기법을 적용
    다음 내용에 따라 모듈을 수정 (또는 수정된 모듈을 반환):
        1) 가지치기 기법에 의해 매개변수 `name` 에 적용된 이진 마스크에 해당하는 명명된 버퍼 `name+'_mask'` 를 추가합니다.
        `name` 파라미터는 가지치기 기법이 적용된 것으로 대체되며, 가지치기 기법이 적용되지 않은
        기존의 파라미터는 `name+'_orig'` 라는 이름의 새로운 매개변수에 저장됩니다.

    인자값:
        module (nn.Module): 가지치기 기법을 적용해야하는 텐서를 포함하는 모듈
        name (string): 모듈 내 가지치기 기법이 적용될 파라미터의 이름

    반환값:
        module (nn.Module): 입력 모듈에 대해서 가지치기 기법이 적용된 모듈

    예시:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    FooBarPruningMethod.apply(module, name)
    return module

######################################################################
# 한번 해봅시다!
model = LeNet()
foobar_unstructured(model.fc3, name='bias')

print(model.fc3.bias_mask)
