"""
텐서보드를 사용한 파이토치 프로파일러
====================================

이 튜토리얼은 모델의 병목 현상을 감시하기위해 텐서보드 플러그인을 파이토치 프로파일 어떻게 사용하는지
보여줍니다.

소개
------------
파이토치 1.8 은 CPU 측 작업 기록 뿐만 아니라
GPU측에서 실행되는 CUDA 커널 기록을 수행할 수 있는 업데이트된 프로파일러 API를 포함하고 있습니다.

프로파일러는 텐서보드 플러그인에 이 정보를 시각화 해줄수 있고 성능 병목들의 분석을 제공해줄 수 있습니다.

이 튜토리얼에서는, 모델 성능 분석을 위한 텐서보드 플러그인을 어떻게 사용하지는 설명하기 위해서
간단한 Resnet 모델을 사용할 것입니다.
Setup
설정
-----
``torch``와``torchvision``을 설치하기 위해서 다음 커맨드를 사용합니다.:
::

   pip install torch torchvision


"""


######################################################################
# 단계
# -----
#
# 1. 데이터와 모델을 준비합니다.
# 2. 프로파일러를 사용하여 실행 이벤트를 기록합니다.
# 3. 프로파일러를 실행합니다.
# 4. 결과를 확인하고 모델 성능 분석을 위해 텐서보드를 사용합니다.
# 5. 프로파일러의 도움으로 성능을 증진시킵니다.
# 6. 다른 고급 기능으로 성능을 분석합니다.
#
# 1. 데이터와 모델을 준비합니다.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 첫번째, 모든 필요한 라이브러리를 import 합니다:
#

import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

######################################################################
# 그 후 입력데이터를 준비한다. 이 튜토리얼에서는 CIFAR10 데이터셋을 사용합니다.
# 원하는 포맷으로 데이터를 변환하고 데이터로더를 사용하여 각각 배치크기로 로드합니다.

transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

######################################################################
# 다음으로, Resnet 모델,손실함수,최적화 객체들을 만듭니다.
# GPU 실행을 위해서, 모델과 로스를 GPU 장치로 이동시킵니다.

device = torch.device("cuda:0")
model = torchvision.models.resnet18(pretrained=True).cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()


######################################################################
# 입력데이터의 각 배치에 대한 훈련 단계를 정의합니다.

def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


######################################################################
# 2. 프로파일러를 사용하여 실행 이벤트를 기록합니다.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 프로파일러는 컨텍스트 매니저를 통해 활성화 되고 가장 유용한 여러 매개변수를 수용합니다:
#
# - ``schedule`` - 호출가능한 스케줄은 단일 매개변수로 단계를 수행하고
#   각 단계로 수행할 프로파일러 작업을 반환합니다.
#   이 예제에서는 다음과 같다. ``wait=1, warmup=1, active=3, repeat=2``,
#   프로파일러는 첫 번째 단계를 건너뛰고,
#   두 번째에서 워밍업을 시작하고,
#   세개의 반복 기록합니다,
#   그 후 트레이스를 사용할 수 있게되고 on_profiler_ready가 호출됩니다.
#   종합적으로,이 싸이클이 두번 반복됩니다. 각 싸이클은 텐서보드 플러그인에서 "span"이라고 불려집니다.
#
#   ``wait`` 단계 동안, 프로파일러는 비활성화 됩니다.
#   ``warmup``단계 동안, 프로파일러가 추적을 시작하지만 결과는 무시됩니다.
#   프로파일링 오버헤드를 줄이기 위한 것입니다.
#   프로파일링 시작 시 발생하는 오버헤드는 크고 프로파일링 결과를 왜곡하기 쉽습니다.
#   ``active`` 단계 동안, 프로파일러는 이벤트를 처리하고 기록합니다.
# - ``on_trace_ready`` - on_trace_ready는 각 단계의 끝에 호출 됩니다.
#   이 예에서는 텐서보드에 대한 결과 파일을 생성하기 위해 ``torch.profiler.tensorboard_trace_handler``를 사용한다.
#   프로파일링 후 결과 파일은 ``.log/resnet18`` 디렉토리에 저장됩니다.
#   TensorBoard에서 프로파일을 분석하기 위해 이 디렉토리를 ``logdir`` 매개변수로 지정합니다.
# - ``record_shapes`` - 연산자 입력의 모양을 기록할지 여부를 나타냅니다.
# - ``profile_memory`` - 트랙 텐서 메모리 할당/할당 해제.
# - ``with_stack`` - 작업에 대한 소스 정보(파일 및 라인 번호) 기록.
#   TensorBoard가 VScode에서 실행되는 경우 (`reference <https://code.visualstudio.com/docs/datascience/pytorch-support#_tensorboard-integration>`_),
#   스택 프레임을 클릭하면 특정 코드 행으로 이동합니다.

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(batch_data)
        prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.


######################################################################
# 3. 프로파일러 실행
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 위의 코드를 실행합니다. 프로파일링 결과는 "..log/resnet18" 디렉토리에 저장됩니다.


######################################################################
# 4. TensorBoard를 사용하여 결과 보기 및 모델 성능 분석
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch 프로파일러 텐서보드 플러그인을 설치합니다.
#
# ::
#
#     pip install torch_tb_profiler
#

######################################################################
# 텐서보드를 시작합니다.
#
# ::
#
#     tensorboard --logdir=./log
#

######################################################################
# Google Chrome 브라우저 또는 Microsoft Edge 브라우저에서 TensorBoard 프로필 URL을 엽니다.
#
# ::
#
#     http://localhost:6006/#pytorch_profiler
#

######################################################################
# 아래와 같이 프로파일러 플러그인 페이지를 볼 수 있습니다.
#
# - Overview
# .. image:: ../../_static/img/profiler_overview1.png
#    :scale: 25 %
#
# 개요에는 모델 성능에 대한 개괄적인 요약이 표시됩니다.
#
# "GPU Summary" 패널에는 GPU 구성 및 GPU 사용량이 표시됩니다.
# 이 예에서는 GPU 사용률이 낮습니다.
# 측정기준의 세부사항은  `여기 <https://github.com/guyang3532/kineto/blob/readme/tb_plugin/docs/gpu_utilization.md>`_.
#
# "Step Time Breakdown"은 각 단계에서 다양한 실행 범주에 걸쳐 소요된 시간의 분포를 보여줍니다.
# 이 예에서는 "DataLoader" 오버헤드가 유의하다는 것을 알 수 있습니다.
#
# 아래의 "성능 권장 사항"에서는 프로파일링 데이터를 사용합니다.
# 발생 가능한 병목 현상을 자동으로 강조하고
# 실행 가능한 최적화 제안을 제공합니다.
#
# 왼쪽 "Views" 드롭다운 목록에서 보기 페이지를 변경할 수 있습니다.
#
# .. image:: ../../_static/img/profiler_views_list.png
#    :alt:
#
#
# - Operator view
# Operator view는 모든 PyTorch 연산자의 성능을 표시합니다.
# Operator view는 호스트 또는 디바이스에서 실행됩니다.
#
# .. image:: ../../_static/img/profiler_operator_view.png
#    :scale: 25 %
# "Self" 기간은 하위 연산자의 시간을 포함하지 않습니다.
# "Total" 기간은 하위 운영자의 시간을 포함합니다.
#
# - View call stack
# 연산자의 "View Callstack "를 클릭하면 이름이 같지만 호출 스택이 다른 연산자가 표시됩니다.
# 그런 다음 이 하위 테이블에서 "View Callstack "를 클릭하면 콜 스택 프레임이 표시됩니다.
#
# .. image:: ../../_static/img/profiler_callstack.png
#    :scale: 25 %
#
# TensorBoard가 VScode 내부에서 실행되는 경우
# (`실행 가이드 <https://devblogs.microsoft.com/python/python-in-visual-studio-code-february-2021-release/#tensorboard-integration>`_),
# 호출 스택 프레임을 클릭하면 특정 코드 줄로 이동합니다.
#
# .. image:: ../../_static/img/profiler_vscode.png
#    :scale: 25 %
#
#
# - Kernel view
# GPU 커널 보기에는 GPU에 걸린 모든 커널의 시간이 표시됩니다.
#
# .. image:: ../../_static/img/profiler_kernel_view.png
#    :scale: 25 %
# SM당 평균 블록 수:
# SM당 블록 수 = 이 커널의 블록 수 / 이 GPU의 SM 수
# 이 숫자가 1보다 작으면 GPU 멀티프로세서가 완전히 사용되지 않음을 나타냅니다.
# "Mean Blocks per SM" 은 각 실행 기간을 가중치로 사용하여 이 커널 이름의 모든 실행의 가중 평균입니다.
#
# Est. Achieved Occupancy 의미:
# Est. Achieved Occupancy 은 이 열의 도구 설명에 정의되어 있습니다.
# 메모리 대역폭 제한 커널과 같은 대부분의 경우 높을수록 좋습니다.
# "Mean Est. Achieved Occupancy"은 각 실행 기간을 가중치로 사용하여 이 커널 이름의 모든 실행의 가중 평균입니다.
#
# - Trace view
# The trace view 는 프로파일링된 연산자와 GPU 커널의 타임라인이 표시됩니다.
# 아래와 같이 선택하여 세부 정보를 볼 수 있습니다.
#
# .. image:: ../../_static/img/profiler_trace_view1.png
#    :scale: 25 %
#
# 오른쪽 도구 모음의 도움을 받아 그래프를 이동하고 확대/축소할 수 있습니다.
# 또한 키보드를 사용하여 시간 표시 막대 내에서 확대/축소하거나 이동할 수 있습니다.
# 'w' 및 's' 키는 마우스 중심에서 확대됩니다.
# 그리고 'a'와 'd' 키를 누르면 시간 표시 막대가 왼쪽과 오른쪽으로 이동합니다.
# 읽을 수 있는 표현이 나타날 때까지 이 키를 여러 번 누를 수 있습니다.
#
# 이 예에서는 ``enumerate(DataLoader)`` 앞에 붙은 이벤트가 많은 시간이 소요됨을 확인할 수 있습니다.
# 그리고 대부분의 기간 동안 GPU는 유휴 상태입니다.
# 이 함수는 호스트측에서 데이터를 로드하고 데이터를 변환하기 때문에,
# 그동안 GPU 리소스가 낭비됩니다.


######################################################################
# 5. 프로파일러의 도움으로 성능을 증진시킨다.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# "Overview" 페이지 하단에 있는 "Performance Recommendation"의 제안에서 병목 현상이 DataLoader임을 암시합니다.
# PyTorch DataLoader는 기본적으로 단일 프로세스를 사용합니다.
# 사용자는 매개 변수 ''num_workers''를 설정하여 다중 프로세스 데이터 로드를 활성화할 수 있습니다.
# `여기 <https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading>`_ 는 좀더 세부사항이다.
#
# 이 예에서는 "Performance Recommendation"에 따라 "num_workers"를 다음과 같이 설정하고,
# ".log/resnet18_4workers"와 같은 다른 이름을 "tensorboard_trace_handler"로 전달한 후 다시 실행합니다.
#
# ::
#
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
#

######################################################################
# 그런 다음 왼쪽 "Runs" 드롭다운 목록에서 최근에 프로파일된 실행을 선택합니다.
#
# .. image:: ../../_static/img/profiler_overview2.png
#    :scale: 25 %
#
# 위의 보기에서, 우리는 이전 실행의 121ms와 비교하여 단계 시간이 약 58ms로 줄어든 것을 발견할 수 있고,
# "DataLoader"의 시간 단축이 주효함을 알 수 있습니다.
#
# .. image:: ../../_static/img/profiler_trace_view2.png
#    :scale: 25 %
#
# 위의 보기에서 "Enumerate(DataLoader)"의 런타임이 줄어드는 것을 볼 수 있고,
#  GPU 활용도가 증가하는 것을 볼 수 있습니다.

######################################################################
# 6. 다른 고급 기능으로 성능을 분석합니다.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# - Memory view
# 프로필 메모리에 "torch.profiler.profile" 인수에 "profile_memory=True"를 추가하십시오.
#
# Note: 파이토치 프로파일러의 실행이 현재 최적화 되지 않았기 때문에,
# ``profile_memory=True`` 를 활성화하는 데는 약 몇분의 시간이 소요 될 것 입니다.
# 시간을 절약하기 위해 먼저 기존 예제를 실행하여 시도해 보십시오.:
#
# ::
#
#     tensorboard --logdir=https://torchtbprofiler.blob.core.windows.net/torchtbprofiler/demo/memory_demo
#
# 프로파일러는 프로파일링 중에 모든 메모리 할당/해제 이벤트를 기록합니다. 
# 모든 특정 연산자에 대해 플러그인은 수명 내에 있는 모든 메모리 이벤트를 집계합니다.
#
# .. image:: ../../_static/img/profiler_memory_view.png
#    :scale: 25 %
#
# "Device" 선택 상자에서 메모리 유형을 선택할 수 있습니다..
# 예를 들어, "GPU0"은 CPU나 다른 GPU를 포함하지 않고 GPU 0에서 각 연산자의 메모리 사용량만 보여주는 표라는 의미이다.
#
# "Size Increase" 모든 할당 바이트를 더하고 모든 메모리 릴리즈 바이트를 뺍니다.
#
# The "Allocation Size"는 메모리 릴리스를 고려하지 않고 모든 할당 바이트를 더합니다.
#
# - Distributed view
# 플러그인은 이제 NCCL을 백엔드로 사용하는 DDP 프로파일링에 대한 분산 보기를 지원합니다.
#
# Azure에서 기존 예제를 사용하여 시도할 수 있습니다.:
#
# ::
#
#     tensorboard --logdir=https://torchtbprofiler.blob.core.windows.net/torchtbprofiler/demo/distributed_bert
#
# .. image:: ../../_static/img/profiler_distributed_view.png
#    :scale: 25 %
#
# "Computation/Communication Overview" 는 계산/통신 비율과 중복 정도를 보여줍니다.
# 이 보기에서, 사용자는 작업자 간의 로드 밸런싱 문제를 파악할 수 있습니다.
# 예를 들어, 한 작업자의 연산 + 중복 시간이 다른 작업자보다 훨씬 클 경우,
# 로드 균형에 문제가 있거나 이 작업자가 스트래글러일 수 있습니다.
#
# "Synchronizing/Communication Overview"는 통신의 효율성을 보여줍니다.
# "Data Transfer Time" 은 실제 데이터 교환 시간입니다.
# "Synchronizing Time" 은 다른 작업자와 대기하고 동기화하는 시간입니다.
#
# 한 작업자의 "Synchronizing Time"이 다른 작업자의 시간보다 훨씬 짧다면,
# 이 작업자는 다른 작업자의 작업량보다 더 많은 계산 작업량을 가질 수 있는 스트래글러일 수 있습니다.
#
# "Communication Operations Stats" 는 각 작업자의 모든 통신 작업에 대한 자세한 통계를 요약합니다.

######################################################################
# 더 배우기
# ----------
#
# 학습을 계속하려면 다음 문서를 살펴보십시오.
# 자유롭게 이슈를 열어보세요. `여기 <https://github.com/pytorch/kineto/issues>`_.
#
# -  `Pytorch TensorBoard Profiler github <https://github.com/pytorch/kineto/tree/master/tb_plugin>`_
# -  `torch.profiler API <https://pytorch.org/docs/master/profiler.html>`_
