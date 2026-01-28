# -*- coding: utf-8 -*-
"""
샘플별 변화도(Per-sample-gradients)
====================
**번역:** `최도윤 <https://github.com/justjs4evr>`_

샘플별 변화도 계산은 데이터 한 배치의 모든 샘플에 대해
변화도를 계산하는 것입니다. 차분 프라이버시(differential privacy), 메타 학습, 최적화 연구에서
유용하게 사용됩니다.

.. note::

   이 튜토리얼은 Pytorch 2.0.0 이상의 버전을 필요로 합니다.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# 다음은 간단한 CNN 모델과 손실 함수입니다.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)


######################################################################
# MNIST 데이터셋을 사용한다고 가정하고, 더미(dummy) 데이터 배치를 만들었습니다.
# 각 데이터는 28x28의 크기를 가지고, 미니배치의 크기를 64로 뒀습니다. 

device = 'cuda'

num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)

targets = torch.randint(10, (64,), device=device)

######################################################################
# 미니배치를 모델에 주고, .backward()를 호출해서
# 변화도를 계산하는 것이 일반적인 모델 학습 방법입니다. 
# 곧, 그 미니배치 전체의 '평균적인' 변화도를 생성할 수 있습니다:

model = SimpleCNN().to(device=device)
predictions = model(data)  # 전체 미니배치를 모델에 통과시킵니다.

loss = loss_fn(predictions, targets)
loss.backward()  # 미니배치의 평균 변화도를 역전파시킵니다.

######################################################################
# 위 방법과는 반대로, 샘플별 변화도 계산은 다음과 같습니다:
#
# - 데이터의 각 샘플에 대해, 순전파와 역전파를 수행하여
#   개별적인(샘플별) 변화도를 구합니다.

def compute_grad(sample, target):
    sample = sample.unsqueeze(0)  # 처리를 위해선 배치 차원을 추가해야 합니다.
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
    """ 각 샘플에 대한 변화도를 직접 계산합니다 """
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

per_sample_grads = compute_sample_grads(data, targets)

######################################################################
# ``sample_grads[0]`` 는 ``model.conv1.weight``에 대한 샘플별 변화도입니다.
# ``model.conv1.weight.shape``는 ``[32, 1, 3, 3]``입니다. 배치 내부에 샘플당 하나씩,
# 총 64개의 변화도가 있다는 점을 주목하시길 바랍니다.

print(per_sample_grads[0].shape)

######################################################################
# 함수 변환(Function Transforms)을 이용한 *효율적인* 샘플별 변화도 계산
# ----------------------------------------------------------------
# 함수 변환을 사용하면 더 효율적으로 샘플별 변화도를 알 수 있습니다.
#
# ``torch.func``의 함수 변환 API는 함수를 대상으로 동작합니다.
# 먼저 손실 함수를 정의하고, 변환해서 
# 샘플별 변화도를 계산하는 함수를 형성하는 것이 전략입니다.
#
# ``torch.func.functional_call`` 함수를 가지고 ``nn.Module``을 함수처럼 다룰 것입니다.
#
# 우선, ``model``의 상태를 매개변수와 버퍼, 두 딕셔너리로 추출해야 합니다.
# 일반적인 PyTorch autograd(예: Tensor.backward(), torch.autograd.grad)를
# 사용하지 않을 것이기 때문입니다.

from torch.func import functional_call, vmap, grad

params = {k: v.detach() for k, v in model.named_parameters()}
buffers = {k: v.detach() for k, v in model.named_buffers()}

######################################################################
# 다음으로, 배치가 아닌 단일 입력에 대한 모델의 손실을 계산하는 함수를 정의합시다.
# 이때 매개변수, 입력, 그리고 목표 변수(target)을 함수가 인자로 받게 하는 것이 중요합니다.
# 그 인자들에 대해 변환을 적용할 것이기 때문입니다.
#
# 참고 - 모델은 배치를 원래 처리하도록 작성되었으니, 
# ``torch.unsqueeze``로 배치 차원을 추가해 줍니다.

def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(model, (params, buffers), (batch,))
    loss = loss_fn(predictions, targets)
    return loss

######################################################################
# 이제, ``grad`` 변환을 사용해서 ``compute_loss``의 첫번째 인자인 
# 매개변수 ``params``에 대해 변화도를 측정하는 새로운 함수를 만들어 봅시다.

ft_compute_grad = grad(compute_loss)

######################################################################
# ``ft_compute_grad``는 단일 (샘플, 목표 변수) 쌍에 대한 변화도를 계산하는 함수입니다.
# ``vmap``을 사용하면 전체 샘플 및 목표 변수들 배치에 대한 변화도를 알 수 있습니다.
# 이때 ``in_dims=(None, None, 0, 0)``로 설정했는데, 이는 데이터와 목표 변수의 
# 0번째 차원에 대해 ``ft_compute_grad``를 매핑하고, 매개변수와 버퍼는
# 각 샘플에 대해 똑같이 사용하기 위함입니다.

ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

######################################################################
# 마지막으로, 변환됨 함수를 이용해 샘플별 변화도를 계산합니다.

ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)

######################################################################
# ``grad``와 ``vmap``을 사용한 결과가 수동으로 하나씩 처리한 결과와 일치하는지 확인합니다:

for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads.values()):
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=1.2e-1, rtol=1e-5)

######################################################################
# 참고: ``vmap``으로 변환할 수 있는 함수의 유형에는 몇 가지 제한이 있습니다. 
# 가장 변환하기 좋은 함수는 순수 함수(pure function)입니다. 즉, 출력이 오직 입력에 
# 의해서만 결정되고 값 수정과 같은 부작용이 없는 함수입니다. 
# ``vmap``은 임의의 파이썬 데이터 구조를 수정하는 작업은 처리할 수 없지만, 
# 많은 PyTorch 인플레이스(in-place) 연산은 처리할 수 있습니다.
#
# 성능 비교
# ----------------------
#
# ``vmap``의 성능이 얼마나 차이 나는지 궁금하신가요?
#
# 현재 A100(Ampere)과 같은 최신 GPU에서 가장 좋은 결과를 얻을 수 있고,
# 이 예제에서 최대 25배의 속도 향상을 확인했습니다. 한편 다음은 빌드 장비에서의 결과입니다:

def get_perf(first, first_descriptor, second, second_descriptor):
    """torch.benchmark 객체들을 받아서 첫번째와 두번째의 차이를 비교합니다"""
    second_res = second.times[0]
    first_res = first.times[0]

    gain = (first_res-second_res)/first_res
    if gain < 0: gain *=-1 
    final_gain = gain*100

    print(f"Performance delta: {final_gain:.4f} percent improvement with {first_descriptor} ")

from torch.utils.benchmark import Timer

without_vmap = Timer(stmt="compute_sample_grads(data, targets)", globals=globals())
with_vmap = Timer(stmt="ft_compute_sample_grad(params, buffers, data, targets)",globals=globals())
no_vmap_timing = without_vmap.timeit(100)
with_vmap_timing = with_vmap.timeit(100)

print(f'Per-sample-grads without vmap {no_vmap_timing}')
print(f'Per-sample-grads with vmap {with_vmap_timing}')

get_perf(with_vmap_timing, "vmap", no_vmap_timing, "no vmap")

######################################################################
# PyTorch에는 이런 방법보다 성능이 뛰어난 다른 최적화 솔루션
# (예: https://github.com/pytorch/opacus)들도 존재합니다. 
# 하지만 ``vmap``과 ``grad``를 조합하는 것만으로 이 정도의 속도 향상을
# 이룰 수 있다는 사실이 멋지지 않나요?
#
# 일반적으로 ``vmap``을 이용한 벡터화는 함수를 반복문에서 실행하는 것보다 빠르고, 
# 수동 배칭(manual batching)과 경쟁할 만한 성능을 보여줍니다. 
# 다만 특정 연산에 대해 ``vmap`` 규칙이 없거나, 하위 커널이 오래된 하드웨어(GPU)에 
# 최적화되지 않은 경우 등에서 예외가 있을 수 있습니다. 이런 경우를 하나라도 발견한다면, 
# GitHub에 이슈를 남겨 알려주시기 바랍니다.