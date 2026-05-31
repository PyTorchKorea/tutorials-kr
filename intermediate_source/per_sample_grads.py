# -*- coding: utf-8 -*-
"""
표본별 변화도(Per-sample gradients)
====================

**번역:** `INHO JEUNG <https://github.com/NamsanMan>`__

표본별 변화도(Per-sample gradients)란
-----------

표본별 변화도(Per-sample gradients) 계산은 데이터 배치에 있는 각 표본의 변화도를 하나씩
계산하는 작업입니다. 이는 차등 개인정보 보호(differential privacy), 메타 학습(meta-learning),
최적화 연구에서 유용하게 쓰이는 값입니다.

.. note::

   이 튜토리얼을 실행하려면 PyTorch 2.0.0 이상이 필요합니다.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# 간단한 CNN과 손실 함수를 정의합니다.

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
# 더미 데이터 배치를 만들고 MNIST 데이터셋으로 작업한다고 가정해 보겠습니다.
# 더미 이미지는 28 x 28 크기이며 크기가 64인 미니 배치를 사용합니다.

device = 'cuda'

num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)

targets = torch.randint(10, (64,), device=device)

######################################################################
# 일반적인 모델 학습에서는 미니 배치를 모델에 전달해 순전파를 수행한 다음 .backward()를
# 호출하여 변화도를 계산합니다. 그러면 전체 미니 배치에 대한 '평균' 변화도가 만들어집니다.

model = SimpleCNN().to(device=device)
predictions = model(data)  # 전체 미니 배치를 모델에 전달합니다.

loss = loss_fn(predictions, targets)
loss.backward()  # 이 미니 배치의 '평균' 변화도를 역전파합니다.

######################################################################
# 위 방식과 달리 표본별 변화도 계산은 다음 과정과 같습니다.
#
# - 데이터의 각 표본에 대해 순전파와 역전파를 수행하여
#   개별 표본의 변화도, 즉 표본별 변화도를 얻습니다.

def compute_grad(sample, target):
    sample = sample.unsqueeze(0)  # 처리를 위해 배치 차원을 앞에 추가합니다.
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
    """각 표본을 직접 처리하여 표본별 변화도를 구합니다."""
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

per_sample_grads = compute_sample_grads(data, targets)

######################################################################
# ``sample_grads[0]``은 model.conv1.weight에 대한 표본별 변화도입니다.
# ``model.conv1.weight.shape``은 ``[32, 1, 3, 3]``입니다.
# 배치의 각 표본마다 변화도가 하나씩 있으므로 총 64개라는 점을 확인할 수 있습니다.

print(per_sample_grads[0].shape)

######################################################################
# 함수 변환으로 표본별 변화도를 *효율적으로* 계산하기
# ------------------------------------------------------
# 함수 변환(function transform)을 사용하면 표본별 변화도를 효율적으로 계산할 수 있습니다.
#
# ``torch.func`` 함수 변환 API는 함수에 변환을 적용합니다.
# 여기서는 먼저 손실을 계산하는 함수를 정의한 다음
# 변환을 적용하여 표본별 변화도를 계산하는 함수를 구성합니다.
#
# ``torch.func.functional_call`` 함수를 사용하여 ``nn.Module`` 을 함수처럼 다룹니다.
#
# 먼저 ``model``의 상태를 parameters와 buffers라는 두 딕셔너리로 추출합니다.
# 일반적인 PyTorch autograd(예: Tensor.backward(), torch.autograd.grad)는 사용하지 않으므로
# 이 값을 분리(detach)합니다.

from torch.func import functional_call, vmap, grad

params = {k: v.detach() for k, v in model.named_parameters()}
buffers = {k: v.detach() for k, v in model.named_buffers()}

######################################################################
# 다음으로 입력 배치가 아니라 단일 인자가 주어졌을 때
# 모델의 손실을 계산하는 함수를 정의하겠습니다.
# 이 함수는 배치 차원이 제거된 단일 인자를 받아야 합니다.
# 변환은 이 인자에 대해 적용할 예정이기 때문입니다.
#
# 참고로 모델은 원래 배치를 처리하도록 작성되었으므로 ``torch.unsqueeze``로
# 배치 차원을 추가합니다.

def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(model, (params, buffers), (batch,))
    loss = loss_fn(predictions, targets)
    return loss

######################################################################
# 이제 ``grad`` 변환을 사용하여 ``compute_loss``의 첫 번째 인자,
# 즉 ``params``에 대한 변화도를 계산하는 새 함수를 만듭니다.

ft_compute_grad = grad(compute_loss)

######################################################################
# ``ft_compute_grad`` 함수는 단일(sample, target) 쌍에 대한 변화도를 계산합니다.
# ``vmap``을 사용하면 표본과 target의 전체 배치에 대해 변화도를 계산할 수 있습니다.
# ``ft_compute_grad``를 data와 targets의 0번째 차원에 매핑하면서
# 각 표본에는 같은 ``params`` 와 buffers를 사용하려 하므로
# ``in_dims=(None, None, 0, 0)``으로 지정합니다.

ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

######################################################################
# 마지막으로 변환된 함수를 사용하여 표본별 변화도를 계산합니다.

ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)

######################################################################
# ``grad``와 ``vmap``을 사용한 결과가
# 각 표본을 직접 하나씩 처리한 결과와 일치하는지 다시 확인할 수 있습니다.

for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads.values()):
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=1.2e-1, rtol=1e-5)

######################################################################
# 간단히 덧붙이면 ``vmap``으로 변환할 수 있는 함수 유형에는 제한이 있습니다.
# 변환하기에 가장 좋은 함수는 순수 함수입니다.
# 순수 함수는 출력이 오직 입력으로만 결정되고 부수 효과(예: 변경)가 없는 함수입니다.
# ``vmap``은 임의의 Python 자료 구조 변경을 처리할 수는 없지만,
# 많은 제자리 PyTorch 연산은 처리할 수 있습니다.
#
# 성능 비교
# ---------
#
# ``vmap``의 성능이 어느 정도인지 궁금할 수 있습니다.
#
# 현재는 A100(Ampere) 같은 최신 GPU에서 가장 좋은 결과를 얻을 수 있으며,
# 이 예제에서는 최대 25배의 속도 향상을 확인했습니다.
# 아래는 빌드 머신에서 얻은 몇 가지 결과입니다.

def get_perf(first, first_descriptor, second, second_descriptor):
    """torch.benchmark 객체를 받아 첫 번째 결과와 두 번째 결과의 차이를 비교합니다."""
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
# PyTorch에서 표본별 변화도를 계산하는 데에는
# https://github.com/pytorch/opacus 같은 다른 최적화된 해법도 있으며,
# 이 방법들 역시 단순한 방법보다 더 좋은 성능을 냅니다.
# 그래도 ``vmap``과 ``grad``를 조합하는 것만으로도
# 꽤 좋은 속도 향상을 얻을 수 있다는 점은 흥미롭습니다.
#
# 일반적으로 ``vmap``을 이용한 벡터화는 함수를 for 루프에서 실행하는 것보다 빠르고,
# 수동 배치 처리와 비교해도 경쟁력 있는 성능을 냅니다.
# 다만 예외도 있습니다. 특정 연산에 대한 ``vmap`` 규칙이 아직 구현되지 않았거나,
# 하위 커널이 오래된 하드웨어(GPU)에 맞게 최적화되지 않은 경우가 그렇습니다.
# 이런 사례를 발견하면 GitHub에 이슈를 열어 알려주세요.
