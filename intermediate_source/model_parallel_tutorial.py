# -*- coding: utf-8 -*-
"""
단일 머신을 사용한 모델 병렬화 모범 사례
===================================================
**저자** : `Shen Li <https://mrshenli.github.io/>`_
**번역** : `안상준 <https://github.com/Justin-A>`_

모델 병렬 처리는 분산 학습 기술에 범용적으로 사용되고 있습니다.
이전 튜토리얼들에서는 여러 GPU를 사용하여 신경망 모델을 학습 시킬 때 어떻게
`DataParallel <https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html>`_ 을 사용하는지에 대해서 살펴보았습니다.
이 방법은 각 GPU에 입력 데이터를 부분적으로 할당하고 동일한 신경망 모델을 복제하여 이용하는 방식이었습니다.
이 방법은 신경망 모델을 상당히 빠르게 학습시킬 수 있는 장점이 있지만, 신경망 모델이 너무 커서 단일 GPU에 할당이 되지 않는 경우에는 동작하지 않습니다.

이번 튜토리얼에서는 ``데이터 병렬 처리`` 가 아닌 **모델 병렬 처리** 문제를 해결하는 방법을 소개합니다.
각 GPU에 모델 전체를 복제하는 것이 아닌, 하나의 모델을 여러 GPU에 분할하여 할당하는 방법입니다.
구체적으로, 10개의 층으로 구성된 ``m`` 신경망 모델에 대해서 ``데이터 병렬 처리`` 방법은 10개의 층을 전부 복제하여 각 GPU에 할당하여 처리하지만,
이와 반대로 2개의 GPU에 모델을 병렬 처리한다면, 각 GPU에 5개의 층씩 각각 할당하여 호스팅할 수 있습니다.

모델 병렬 처리의 전반적인 아이디어는 모델의 서브 네트워크들을 각각 다른 GPU에 할당하고,
각 장비 별로 순전파를 진행하여 계산되는 출력값들을 각 장비 간 공유하여 이용하는 것입니다.
이용하고자 하는 신경망 모델을 부분적으로 각 GPU에 할당하는 것이기 때문에, 여러 GPU를 이용하여 더 큰 신경망 모델을 할당하고 학습시킬 수 있습니다.
이번 튜토리얼은 거대한 모델을 제한된 수의 GPU에 분할하여 할당하지 않고, 그 대신, 모델 병렬 처리의 아이디어를 이해하는 목적으로 작성되었습니다.
모델 병렬 처리의 아이디어를 활용하여 실제 어플리케이션에 적용하는 것은 여러분의 몫입니다.

.. note::
   신경망 모델을 여러 서버를 이용하여 학습시키는 병렬 학습 방법에 대해서는 다음 튜토리얼을 참고하세요.
   `분산 프레임워크 RPC 시작해보기 <rpc_tutorial.html>`__

Basic Usage
-----------
"""

######################################################################
# 2개의 층으로 이루어진 간단한 신경망 모델을 이용해서 기본적인 내용을 실습해봅시다.
# 신경망 모델을 2개의 GPU에 할당하여 실행하기 위해서, 각 1개의 층을 각각 다른 GPU에 할당하고,
# 입력 텐서값과 중간 산출물 텐서값을 신경망 모델의 구성에 맞게 배치합니다.
#

import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0') # 첫 번째 층을 첫 번째 GPU에 할당
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')  # 두 번째 층을 두 번째 GPU에 할당

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1')) # 첫 번째 층의 산출물을 두 번째 GPU에 할당하여 진행

######################################################################
# 위의 ``ToyModel`` 예제는 선형 층과 텐서 값을 4개의 ``to(device)`` 장비에 적절하게 할당하는 것이 아닌,
# 단일 GPU로 신경망 모델을 구현하는 것과 매우 유사한 구조인 것임을 확인할 수 있습니다.
# 다시 말해, GPU에 텐서 값 혹은 층을 할당하는 것 외에는 추가적으로 설정하는 부분이 없습니다.
# ``backward()`` 와 ``torch.optim`` 코드를 통해 단일 GPU를 이용하여 신경망 모델의 가중치 값을 업데이트하는 것처럼, 자동으로 오차에 의한 기울기 값을 반영합니다.
# 여러분은 레이블값과 신경망 모델의 최종 출력 텐서 값을 이용하여 오차를 계산할 수 있도록 동일한 GPU에 할당하는 것만 주의하면 됩니다.

model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1') # 신경망 모델의 최종 출력값과 동일한 GPU에 할당
loss_fn(outputs, labels).backward()
optimizer.step()

######################################################################
# 기존에 존재하는 모듈에 모델 병렬 처리 적용해보기
# ---------------------------------------------------
#
# 기존에 단일 GPU에 존재하는 모듈을 여러 GPU에 할당하는 것은 단지 몇 줄의 코드를 수정하는 것으로도 쉽게 가능합니다.
# 아래에 있는 코드들은 ``torchvision.models.reset50()`` 모델을 2개 GPU로 분할하는 방법입니다.
# 이 아이디어는, 기존에 존재하는 ResNet 모듈을 상속받아 설계할 때, 2개의 GPU에 층을 나누어 설계하는 방식으로 진행됩니다.
# 그 후, 2개 GPU에서 계산되는 중간 산출물 텐서값을 적절히 배치하기 위헤 순전파 메소드를 수정합니다.



from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0') # 첫 번째 GPU에 일련의 과정을 할당

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1') # 두 번째 GPU에 일련의 과정을 할당

        self.fc.to('cuda:1') # ResNet50 구성요소를 두 번째 GPU에 할당

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1')) # seq1의 출력값을 두 번쨰 GPU에 할당하여 연결
        return self.fc(x.view(x.size(0), -1))


######################################################################
# 위의 예제에서는 단일 GPU에 신경망 모델을 할당하여 학습시키기에는 모델 크기가 너무 클 때 발생하는 문제를 해결하는 방법입니다.
# 하지만, 여러분은 단일 GPU를 이용할 때보다 학습 과정이 오래걸리며, 이는 여러분들이 이미 알고 있는 내용이었을 수 있습니다.
# 그 이유는, 두 개의 GPU가 동시에 계산하는 것이 아니라 1개의 GPU는 계산하지 않고 대기하고 있기 때문입니다.
# 또한, 두 번째 층 (layer2)이 할당된 첫 번째 GPU에서 계산된 결과를 세 번째 층 (layer3)이 할당된 두 번째 GPU로 텐서값을 복사하기 때문에 계산 과정이 더 길어지게 됩니다.
#
# 코드 실행 시간을 정량적으로 살펴보기 위해 실험을 하나 해봅시다. 입력 텐서값과 레이블값을 랜덤으로 설정한 후,
# 이미 존재하는 ``torchvision.models.resnet50()`` 과, 모델 병렬 처리를 진행한 ``ModelParallelResNet50`` 을 통해 학습을 진행합니다.
# 학습 진행을 완료한 후, 두 모델들은 랜덤으로 생성된 데이터로 학습을 진행했기 때문에 실용적인 예측을 하진 못하지만, 학습 진행 시간을 실용적으로 비교하여 할 수 있습니다.


import torchvision.models as models

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for _ in range(num_batches):
        # 입력 텐서값과 레이블값을 랜덤으로 생성합니다.
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # 입력값을 이용하여 순전파를 진행합니다.
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # 역전파를 진행하여 신경망 모델의 가중치를 업데이트합니다.
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()


######################################################################
# 위에서 정의한 ``train(model)`` 메소드는 nn.MSELoss (Mean Squared Error ; 평균 제곱 오차) 로 손실 함수를 정의하여 신경망 모델을 학습하는 것을 의미합니다.
# 그리고, ``optim.SGD`` 메소드는 최적화 방식을 의미합니다. 위 방식은 128 * 128 크기의 이미지가 120개로 구성된 배치 데이터가 3개 존재하는 상황을 모방하기 위해 랜덤으로 생성하였습니다.
# 그리고나서, 우리는 ``timeit`` 을 이용하여 ``train(model)`` 메소드를 10회 실행하여 학습을 진행하고, 학습 실행 시간에 대해서 표준 편차값을 반영하는 이미지를 생성하여 저장합니다.


import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit

num_repeat = 10

stmt = "train(model)"

setup = "model = ModelParallelResNet50()"
mp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
rn_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


plot([mp_mean, rn_mean],
     [mp_std, rn_std],
     ['Model Parallel', 'Single GPU'],
     'mp_vs_rn.png')


######################################################################
#
# .. figure:: /_static/img/model-parallel-images/mp_vs_rn.png
#    :alt:
#

# 실험 결과, 모델 병렬 철리하여 학습하는 시간이 단일 GPU로 학습하는 시간보다 약 7% ``4.02/3.75-1=7%``정도
# 오래 걸리는 것을 확인할 수 있습니다. 그러므로, 순전파와 역전파를 진행하면서 GPU 간 텐서값들이
# 복제되어 이용하는 시간이 약 7%정도 소요되는 것으로 결론지을 수 있습니다. 학습하는 과정 속에서
# 2개의 GPU 중 1개의 GPU가 계산하지 않고 대기하고 있기 때문에, 이를 해결하여
# 학습 시간을 빠르게 개선시킬 수 있습니다. 그 중 한 가지 방법은, 학습 단위인 미니 배치 1개의 데이터를
# 2개로 분할하는 파이프라인을 생성하여, 분할된 첫 번째 데이터가 첫 번째 층을 통과하여 두 번째 층으로
# 복제되고, 두 번째 층을 통과할 때, 두번재로 분할된 데이터가 첫 번쨰 층을 통해 계산되는 방식으로 설정하는 것입니다.
# 이러한 방법을 통해서 2개의 GPU가 2개로 분할된 데이터를 동시에 처리할 수 있으며 학습 시간을 단축시킬 수 있습니다.

######################################################################
# 입력 텐서값을 분할하는 파이프라인을 설계하여 학습 시간을 단축하는 방법에 대한 예제
# ---------------------------------------------------------------------------------------
#
# 아래에 있는 실험은, 120개의 이미지로 구성된 1개의 미니 배치 데이터를 20개씩 나누어 진행하는
# 과정입니다. 아래의 과정을 실행할 때, PyTorch가 CUDA 연산을 비동기적으로 이용하기 때문에,
# 프로세스를 실행하는 스레드를 여러개 생성할 필요가 없습니다.


class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. s_prev는 두 번째 GPU에서 실행됩니다.
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next는 A.와 동시에 진행되면서 첫 번째 GPU에서 실행됩니다.
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)


setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

plot([mp_mean, rn_mean, pp_mean],
     [mp_std, rn_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')

######################################################################
# GPU 간 텐서값이 복사되는 것은 현재 계산되고 있는 소스값과, 소스값의 목적지 GPU 간 연산되고 있는
# 스트림과 동기화되는 것을 주의하세요. 만약 여러 스트림을 생성하여 진행하고 있다면, GPU 간 텐서값이
# 정상적으로 복사되어 계산되고 있는지 꼭 확인해야 합니다. 만약 복사되는 과정 중에 소스값을 이용하거나,
# GPU의 텐서값을 읽거나 쓰는 것은 올바르게 계산되지 않을 수 있습니다. 위의 예제에서는 소스값 및 GPU
# 텐서값을 기본 스트림만 이용하여 진행하므로 추가적인 동기화 과정을 진행할 필요는 없습니다.

######################################################################
#
# .. figure:: /_static/img/model-parallel-images/mp_vs_rn_vs_pp.png
#    :alt:
#
# 파이프라인을 이용하여 미니 배치 내 데이터를 분할하여 적용하였을 때, ResNet50 신경망 모델의
# 학습 시간이 약 49% ``3.75/2.51-1=49%`` 정도 단축된 것을 이번 실험을 통해 확인할 수 있습니다. 하지만, 이상적으로
# 학습 시간이 2배 단축되는 것에 비해 다소 적게 학습 시간이 단축되었습니다. 파이프라인을 이용할 때,
# ``split_sizes`` 매개변수를 도입하였기 때문에, 파이프라인을 이용하는 것이 학습 시간 단축에 얼마나
# 영향을 미쳤는지 불분명합니다. 직관적으로 생각하였을 때, ``split_sizes`` 매개변수 값을 작게 설정한다면,
# 아주 소규모의 CUDA 연산이 많이 진행되고, ``split_sizes`` 매개변수 값을 크게 설정한다면, 첫 번째와
# 마지막 분리될 때 비교적 긴 시간 동안 CUDA 연산이 이루어지게 됩니다. 둘 다 최적의 설정이 아닙니다.
# 따라서, ``split_sizes`` 매개변수 값을 최적으로 설정하였을 때, 학습 시간 과정이 단축될 수 있을 것이라
# 기대됩니다. ``split_sizes`` 매개변수 값을 조정하여 실험하면서 최적의 값을 찾아봅시다.


means = []
stds = []
split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60]

for split_size in split_sizes:
    setup = "model = PipelineParallelResNet50(split_size=%d)" % split_size
    pp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    means.append(np.mean(pp_run_times))
    stds.append(np.std(pp_run_times))

fig, ax = plt.subplots()
ax.plot(split_sizes, means)
ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
ax.set_ylabel('ResNet50 Execution Time (Second)')
ax.set_xlabel('Pipeline Split Size')
ax.set_xticks(split_sizes)
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig("split_size_tradeoff.png")
plt.close(fig)

######################################################################
#
# .. figure:: /_static/img/model-parallel-images/split_size_tradeoff.png
#    :alt:
#
# 실험 결과, ``split_size`` 매개변수값을 12로 설정하였을 때, 학습 시간이 54% 수준으로
# 가장 많이 단축되었습니다. 아직 학습 시간을 더 단축시킬 수 있는 방법은 다양하게 존재합니다.
# 예를 들어, 첫 번째 GPU에서 모든 연산과정이 기본으로 설정되어 진행됩니다. 이는 미니배치 분할 과정 중,
# 현재 진행되는 과정의 다음 단계는 현재 진행되는 과정과 동시에 복제가 이루어질 수 없는 것을 의미합니다.
# 그러나, 이전과 다음 단계의 분할과정이 다른 텐서값을 이용하기 때문에, 다른 계산과 중복되어 진행되어도
# 문제가 없습니다. 이에 대해서, 2개 GPU에 여러개의 스트림을 사용하는 것이 필요하며, 서로 다른 서브 네트워크
# 구조가 서로 다른 스트림을 관리하는 전략이 요구됩니다. 모델 병렬 처리에 대해서 여러 스트림을 사용하는 방법이
# 일반적을로 존재하지 않기 때문에 이번 튜토리얼에서는 설명하지 않습니다.

"""
.. note::
   이번 게시물에서는 다양한 성능 측정값을 확인할 수 있습니다. 여러분은 위의 예제를 실행할 때 마다 매번
   다른 결과를 확인할 수 있습니다. 그 이유는, 이용하는 소프트웨어 및 하드웨어에 따라 결과가
   다르게 나타나기 때문입니다. 여러분이 이용하고 있는 환경 내에서 가장 좋은 성능을 얻기 위해서는, 곡선을 그려서
   최적의 ``split_size`` 값을 도출한 후, 해당 값을 이용하여 미니 배치 내 데이터를 분리하는 파이프라인을
   생성하는 것입니다.
"""
