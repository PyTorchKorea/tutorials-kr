# -*- coding: utf-8 -*-
"""
모델 앙상블
================

**번역**: `조형서 <https://github.com/ChoHyoungSeo/>`_

본 튜토리얼에서는 ``torch.vmap`` 을 활용하여 모델 앙상블을 벡터화하는 방법을 설명합니다.

모델 앙상블이란?
-------------------------
모델 앙상블은 여러 모델의 예측값을 함께 결합하는 것을 의미합니다.
일반적으로 이 작업은 일부 입력값에 대해 각 모델을 개별적으로 실행한 다음 예측을 결합하는 방식으로 실행됩니다.
하지만 동일한 아키텍처로 모델을 실행하는 경우, ``torch.vmap`` 을 활용하여 함께 결합할 수 있습니다.
``vmap`` 은 입력 tensor의 여러 차원에 걸쳐 함수를 매핑하는 함수 변환입니다. 이 함수의
사용 사례 중 하나는 for 문을 제거하고 벡터화를 통해 속도를 높이는 것입니다.

간단한 MLP 앙상블을 활용하여 이를 수행하는 방법을 살펴보겠습니다.

.. note::
   이 튜토리얼의 실행을 위해서는 PyTorch 2.0 또는 이상의 버전이 필요합니다.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# 다음은 간단한 MLP 입니다.
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

######################################################################
# 더미 데이터를 생성하고 MNIST 데이터 셋으로 작업한다고 가정해 보겠습니다.
# 따라서 이미지는 28x28 사이즈이며 미니 배치 크기는 64입니다.
# 더 나아가 10개의 서로 다른 모델에서 나온 예측값을 결합하고 싶다고 가정해 보겠습니다.

device = 'cuda'
num_models = 10

data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)

models = [SimpleMLP().to(device) for _ in range(num_models)]

######################################################################
# 예측값을 생성하는 데는 몇 가지 옵션이 있습니다.
# 각각의 모델에 다른 무작위 미니 배치 데이터를 줄 수 있고
# 각각의 모델에 동일한 미니 배치의 데이터를 줄 수 있습니다.
# (예를 들어, 다른 모델 초기값의 영향을 테스트할 경우)

######################################################################
# 옵션 1: 각각의 모델에 다른 미니 배치를 주는 경우

minibatches = data[:num_models]
predictions_diff_minibatch_loop = [model(minibatch) for model, minibatch in zip(models, minibatches)]

######################################################################
# 옵션 2: 같은 미니 배치를 주는 경우

minibatch = data[0]
predictions2 = [model(minibatch) for model in models]

######################################################################
# ``vmap`` 을 활용하여 앙상블 벡터화하기
# -------------------------------------------
#
# ``vmap`` 을 사용하여 for 문의 속도를 높여보겠습니다. 먼저 ``vmap`` 과 함께 사용할 모델을 준비해야 합니다.
#
#
# 먼저, 각 매개변수를 쌓아 모델의 상태를 결합해 보겠습니다.
# 예를 들어, ``model[i].fc1.weight`` 의 shape은 ``[784, 128]`` 입니다.
# 이 10개의 모델 각각에 대해 ``.fc1.weight`` 를 쌓아 ``[10, 784, 128]`` shape의 큰 가중치를 생성할 수 있습니다.
#
# 파이토치에서는 이를 위해 ``torch.func.stack_module_state`` 라는 함수를 제공하고 있습니다.
#
from torch.func import stack_module_state

params, buffers = stack_module_state(models)

######################################################################
# 다음으로, ``vmap`` 에 대한 함수를 정의해야 합니다. 이 함수는 파라미터, 버퍼, 입력값이 주어지면 모델을 실행합니다.
# 여기서는 ``torch.func.functional_call`` 을 활용하겠습니다.

from torch.func import functional_call
import copy

# 모델 중 하나의 "stateless" 버전을 구축합니다.
# "stateless"는 매개변수가 메타 tensor이며 저장소가 없다는 것을 의미합니다.
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')

def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))

######################################################################
# 옵션 1: 각 모델에 대해 서로 다른 미니 배치를 활용하여 예측합니다.
#
# 기본적으로, ``vmap`` 은 모든 입력의 첫 번째 차원에 걸쳐 함수에 매핑합니다.
# ``stack_module_state`` 를 사용하면 각 ``params`` 와 버퍼는 앞쪽에 'num_models'
# 크기의 추가 차원을 가지며, 미니 배치는 'num_models' 크기가 됩니다.

print([p.size(0) for p in params.values()]) # 선행 'num_models' 차원 표시

assert minibatches.shape == (num_models, 64, 1, 28, 28) # 미니 배치의 선행 차원이 'num_models' 크기인지 확인합니다.

from torch import vmap

predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)

# ``vmap`` 예측이 맞는지 확인합니다.
assert torch.allclose(predictions1_vmap, torch.stack(predictions_diff_minibatch_loop), atol=1e-3, rtol=1e-5)

######################################################################
# 옵션 2: 동일한 미니 배치 데이터를 활용하여 예측합니다.
#
# ``vmap`` 에는 매핑할 차원을 지정하는 ``in_dims`` 라는 인자가 있습니다.
# ``None`` 을 사용하면 10개 모델에 모두 동일한 미니 배치를 적용하도록
# ``vmap`` 에 알려줄 수 있습니다.

predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)

assert torch.allclose(predictions2_vmap, torch.stack(predictions2), atol=1e-3, rtol=1e-5)

######################################################################
# 참고 사항: ``vmap`` 으로 변환할 수 있는 함수 유형에는 제한이 있습니다.
# 변환하기에 가장 좋은 함수는 입력값에 의해서만 출력이 결정되고
# 다른 부작용 (예. 변이) 이 없는 순수 함수(pure function) 입니다.
# ``vmap`` 은 임의의 변이된 파이썬 자료구조는 처리할 수 없지만,
# 다양한 내장된 파이토치 연산은 처리할 수 있습니다.

######################################################################
# 성능
# -----------
# 성능 수치가 궁금하신가요? 수치는 다음과 같습니다.

from torch.utils.benchmark import Timer
without_vmap = Timer(
    stmt="[model(minibatch) for model, minibatch in zip(models, minibatches)]",
    globals=globals())
with_vmap = Timer(
    stmt="vmap(fmodel)(params, buffers, minibatches)",
    globals=globals())
print(f'Predictions without vmap {without_vmap.timeit(100)}')
print(f'Predictions with vmap {with_vmap.timeit(100)}')

######################################################################
# ``vmap`` 을 사용하면 속도가 크게 향상됩니다!
#
# 일반적으로, ``vmap`` 을 사용한 벡터화는 for 문에서 함수를 실행하는 것보다
# 빠르며 수동 일괄 처리와 비슷한 속도를 냅니다. 하지만 특정 연산에 대해 ``vmap`` 규칙을
# 구현하지 않았거나 기본 커널이 구형 하드웨어(GPUs)에 최적화되지 않은 경우와 같이
# 몇 가지 예외가 있습니다. 이러한 경우가 발견되면, GitHub에 이슈를 생성해서 알려주시기 바랍니다.
