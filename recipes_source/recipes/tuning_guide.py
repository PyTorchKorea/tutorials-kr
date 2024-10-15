"""
성능 튜닝 가이드
*************************
**저자**: `Szymon Migacz <https://github.com/szmigacz>`_
**역자**: `오왕택 <https://github.com/ohkingtaek>`_

성능 튜닝 가이드는 PyTorch에서 딥러닝 모델의 학습이나 추론 속도를 향상시킬 수 있는 최적화 기법과 같은 좋은 예시를 소개합니다.
제시된 기법은 몇 줄의 코드만 변경해서 구현 가능하며, 모든 도메인의 다양한 딥러닝 모델에 적용할 수 있습니다.

일반적인 최적화 기법
---------------------
"""

###############################################################################
# 비동기식으로 데이터 가져오기 및 데이터 증강법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
# 는 각각 워커의 subprocess에서 비동기식 데이터 로딩과 데이터 증강을 지원합니다. 
# ``DataLoader`` 의 num_worker 기본 설정은 ``num_worker=0`` 으로, 이는 데이터 로딩이 
# 동기적으로 이루어지며 메인 프로세스에서 실행됨을 의미합니다. 결과적으로 메인 학습 프로세스는 데이터를 
# 사용할 수 있을 때까지 기다려야 실행할 수 있습니다.
#
# ``num_workers > 0`` 으로 설정하면 비동기식 데이터 로딩과 학습과 데이터 로딩의 동시 처리가 
# 가능합니다. ``num_workers`` 값은 작업량, CPU, GPU, 학습 데이터의 위치에 따라 조정해야 
# 합니다.
#
# ``DataLoader`` 는 ``pin_memory`` 인자를 받으며 기본값은 ``False`` 입니다. GPU를 
# 사용하는 경우 ``pin_memory=True`` 로 설정하는 것이 좋습니다. 이는 ``DataLoader`` 가 
# 고정된 메모리를 사용하고, 호스트에서 GPU로 더 빠르고 비동기적인 메모리 복사합니다.

###############################################################################
# 검증 및 추론 시 변화도 계산 비활성화하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch는 변화도가 필요한 Tensor와 관련된 모든 연산의 중간 버퍼를 저장합니다. 하지만 일반적으로 
# 검증이나 추론 단계에서는 변화도가 필요하지 않습니다.  
# `torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad>`_
# 컨텍스트 관리자를 사용하여 특정 코드 블록 내에서 변화도 계산을 비활성화할 수 있습니다. 
# 이를 통해 실행 속도가 빨라지고 필요한 메모리 양이 줄어듭니다.
# `torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad>`_
# 는 함수 데코레이터로도 사용할 수 있습니다.

###############################################################################
# 합성곱 계층 이후에 바로 배치 정규화 계층이 오는 경우에 편향을 비활성화하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.nn.Conv2d() <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_ 
# 함수에는 기본적으로 ``bias`` 매개변수가 ``True`` 로 설정되어 있습니다 (이는 
# `Conv1d <https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d>`_ 
# 및 
# `Conv3d <https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d>`_
# 에서도 동일합니다).
#
# ``nn.Conv2d`` 계층 바로 뒤에 ``nn.BatchNorm2d`` 계층이 이어진다면 합성곱 계층에서 
# 편향은 필요하지 않으므로 ``nn.Conv2d(..., bias=False, ...)`` 로 설정하세요. 
# ``BatchNorm2d`` 의 첫 단계에서 평균을 빼주기 때문에 필요하지 않으며, 이는 편향의 효과를 
# 상쇄시킵니다.
#
# 이 원리는 1차원 및 3차원 합성곱에서도 동일하게 적용되며 ``BatchNorm`` (또는 다른 정규화 
# 계층)이 합성곱의 편향과 동일한 차원을 정규화할 경우에 해당됩니다.
#
# `torchvision <https://github.com/pytorch/vision>`_ 에서 제공하는 모델은 
# 이미 이 최적화를 구현하고 있습니다.

###############################################################################
# model.zero_grad()나 optimizer.zero_grad() 대신 parameter.grad = None 사용하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 다음과 같은 방식으로 변화도를 초기화하는 대신:
model.zero_grad()
# 또는
optimizer.zero_grad()

###############################################################################
# 아래와 같은 방법을 대신 사용하세요:

for param in model.parameters():
    param.grad = None

###############################################################################
# 두 번째 코드는 각 개별 매개변수의 메모리를 0으로 초기화하지 않으며, 
# 이후의 역전파 과정에서 변화도를 저장할 때 더하기 대신 대입 연산을 사용하여 메모리 연산 수를 줄입니다.
#
# 변화도를 0으로 설정하는 것과 ``None`` 으로 설정하는 것은 약간의 수치적 차이가 있으므로 자세한 
# 내용은 
# `torch.optim <https://pytorch.org/docs/master/optim.html#torch.optim.Optimizer.zero_grad>`_
# 를 참조하세요.
#
# 또는 PyTorch 1.7부터 ``model`` 이나 ``optimizer.zero_grad(set_to_none=True)``
# 를 호출할 수 있습니다.

###############################################################################
# 연산을 결합하여 최적화하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 행렬에서 element-wise 덧셈, 곱셈 같은 연산과 `sin()` , `cos()` , `sigmoid()` 같은 수학 
# 함수 등의 point-wise 연산들은 하나의 커널로 결합할 수 있습니다. 이러한 결합은 메모리 접근과 커널 
# 실행 시간을 줄이는 데 도움이 됩니다. 일반적으로 point-wise 연산은 메모리에 바인딩됩니다.
# PyTorch의 eager-mode에서는 각 연산마다 커널을 실행하므로 메모리에서 데이터를 불러와 연산을 
# 수행하고 (종종 가장 시간이 적게 걸리는 단계) 결과를 다시 메모리에 쓰는 과정이 필요합니다.
#
# 결합된 연산자를 사용하면 여러 point-wise 연산을 위해 단 하나의 커널만 실행되고, 데이터는 한 
# 번만 불러오고 저장됩니다. 특히 이러한 효율적인 방법은 활성화 함수, 옵티마이저, 직접 수정한 RNN 셀 
# 등에서 유용합니다.
#
# PyTorch 2에서는 TorchInductor라는 컴파일러를 통해 자동으로 커널을 결합하는 compile-mode를 
# 도입했습니다. TorchInductor는 단순한 element-wise 연산뿐만 아니라 최적의 성능을 위해 
# point-wise 연산과 축소(reduction) 연산을 고급 결합할 수 있는 기능을 제공하여 성능을 
# 최적화합니다.
#
# 가장 간단한 경우, 연산 결합은 함수 정의에 
# `torch.compile <https://pytorch.org/docs/stable/generated/torch.compile.html>`_ 
# 데코레이터를 적용하여 활성화할 수 있습니다. 예시:

@torch.compile
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

###############################################################################
# 고급 사용 사례에 대해서는 `Introduction to torch.compile 
# <https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html>`_
# 를 참조하세요.

###############################################################################
# 컴퓨터 비전 모델에 대해 channels_last 메모리 형식 활성화하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch 1.5에서는 합성곱 신경망에 대해 channels_last 메모리 형식을 지원하기 시작했습니다. 
# 이 포맷은 `Tensor Cores <https://www.nvidia.com/en-us/data-center/tensor-cores/>`_
# 를 사용하여 합성곱 신경망을 더욱 가속화하기 위해 
# `AMP <https://pytorch.org/docs/stable/amp.html>`_
# 와 함께 사용할 수 있도록 설계되었습니다.
#
# ``channels_last`` 기능은 아직 실험 단계에 있지만 표준 컴퓨터 비전 모델(예시: ResNet-50, 
# SSD)에서는 동작할 것으로 예상됩니다. 모델을 ``channels_last`` 형식으로 변환하는 방법에 대해서는 
# `(베타) PyTorch를 사용한 Channels Last 메모리 형식 <https://tutorials.pytorch.kr/intermediate/memory_format_tutorial.html>`_
# 을 참조하세요. 튜토리얼에는
# `기존 모델들 변환하기 <https://tutorials.pytorch.kr/intermediate/memory_format_tutorial.html#converting-existing-models>`_
# 섹션이 포함되어 있습니다.

###############################################################################
# 중간 버퍼를 체크포인트로 만드는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 버퍼 체크포인트 저장은 모델 학습 중 메모리 용량 부담을 완화하기 위한 기법입니다. 역전파에서 앞부분의 
# 변화도를 계산하기 위해 모든 계층의 입력을 저장하는 대신, 일부 계층의 입력만 저장하고 나머지는
# 역전파 중에 재계산합니다. 메모리 요구 사항이 줄어들어 배치 크기를 증가시킬 수 있으며, 이는 활용 
# 효율을 개선할 수 있습니다.
#
# 체크포인트 저장할 대상은 신중하게 선택해야 합니다. 가장 좋은 방법은 재계산 비용이 적은 대규모 
# 레이어의 출력을 저장하지 않는 것입니다. 예를 들어, 활성화 함수(예시: ``ReLU`` , ``Sigmoid`` 
# , ``Tanh`` ), up/down 샘플링, 작은 누적 깊이(accumulation depth)를 가진 행렬-벡터 연산 
# 등이 체크포인트 저장 대상으로 적합합니다.
#
# PyTorch는 자동으로 체크포인트 저장 및 재계산을 수행하는 
# `torch.utils.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_ 
# API를 지원합니다.

###############################################################################
# 디버깅 API 비활성화
# ~~~~~~~~~~~~~~~~~~~~~~
# 많은 PyTorch API는 디버깅을 위해 설계되었으며 정규 학습 실행 시에는 비활성화해야 합니다.
#
# * 이상탐지 (anomaly detection):
#   `torch.autograd.detect_anomaly <https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly>`_
#   또는
#   `torch.autograd.set_detect_anomaly(True) <https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_detect_anomaly>`_
# * profiler 관련:
#   `torch.autograd.profiler.emit_nvtx <https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx>`_,
#   `torch.autograd.profiler.profile <https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile>`_
# * autograd ``gradcheck``:
#   `torch.autograd.gradcheck <https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradcheck>`_
#   또는
#   `torch.autograd.gradgradcheck <https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradgradcheck>`_
#

###############################################################################
# CPU 관련 최적화
# --------------------------

###############################################################################
# 비균일 메모리 접근(NUMA) 제어 활용 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NUMA(비균일 메모리 접근)는 다중 메모리 컨트롤러와 블록이 있는 멀티 소켓 머신에서 메모리의 지역성을 
# 활용하기 위해 데이터 센터 머신에서 사용되는 메모리 레이아웃 디자인입니다. 일반적으로 딥러닝 작업, 
# 학습 또는 추론 모두에서 NUMA 노드 간의 하드웨어 자원 접근 없이 더 나은 성능을 발휘합니다. 따라서 
# 추론은 각 인스턴스가 하나의 소켓에서 실행되도록 여러 인스턴스로 실행할 수 있으며, 이를 통해 처리량을 
# 증가시킬 수 있습니다. 단일 노드에서의 학습 작업에는 분산 학습이 권장되며, 이를 통해 각 학습 
# 프로세스가 하나의 소켓에서 실행되도록 할 수 있습니다.
#
# 다음 명령어는 N번째 노드의 코어에서만 PyTorch 스크립트를 실행하며, 소켓 간 메모리 
# 접근을 피하여 메모리 접근 오버헤드를 줄입니다.
#
# .. code-block:: sh
#
#    numactl --cpunodebind=N --membind=N python <pytorch_script>

###############################################################################
# 자세한 설명은 
# `여기 <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html>`_
# 서 확인할 수 있습니다.

###############################################################################
# OpenMP 활용하는 방법
# ~~~~~~~~~~~~~~~~~
# OpenMP는 병렬 계산 작업의 성능을 향상시키기 위해 사용됩니다.
# ``OMP_NUM_THREADS`` 는 계산 속도를 높이는 가장 간단한 환경 변수입니다. 이는 OpenMP 계산에 
# 사용되는 스레드 수를 결정합니다.
# CPU Affinity 설정은 작업이 여러 코어에 분배되는 방식을 제어합니다. 이는 통신 오버헤드와 캐시 라인 
# 무효화 오버헤드 또는 페이지 스레싱 등에 영향을 미치므로 CPU 친화도를 적절히 설정하면 성능 향상이 
# 가능합니다. ``GOMP_CPU_AFFINITY`` 와 ``KMP_AFFINITY`` 는 OpenMP 스레드를 물리적 처리 
# 장치에 바인딩하는 방법을 결정합니다. 자세한 정보는 
# `여기 <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html>`_
# 에서 확인할 수 있습니다.

###############################################################################
# 다음 명령어를 사용하면 PyTorch가 N개의 OpenMP 스레드에서 작업을 실행합니다.
#
# .. code-block:: sh
#
#    export OMP_NUM_THREADS=N

###############################################################################
# 일반적으로 GNU OpenMP 구현에서 CPU 친화도를 설정하기 위해 다음 환경 변수를 사용합니다. 
# ``OMP_PROC_BIND`` 는 스레드가 프로세서 간에 이동할 수 있는지 여부를 지정합니다. 이를 CLOSE로 
# 설정하면 OpenMP 스레드가 기본 스레드에 가까운 위치 파티션에 유지됩니다. ``OMP_SCHEDULE`` 는 
# OpenMP 스레드가 어떻게 스케줄링 되는지를 결정합니다. ``GOMP_CPU_AFFINITY`` 는 스레드를 특정 
# CPU에 바인딩합니다.
#
# .. code-block:: sh
#
#    export OMP_SCHEDULE=STATIC
#    export OMP_PROC_BIND=CLOSE
#    export GOMP_CPU_AFFINITY="N-M"

###############################################################################
# Intel OpenMP 런타임 라이브러리(``libiomp``)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 기본적으로 PyTorch는 병렬 계산을 위해 GNU OPENMP(GNU ``libgomp``)를 사용합니다. Intel 
# 플랫폼에서는 Intel OpenMP 런타임 라이브러리(``libiomp``)가 OpenMP API 사양 지원을 
# 제공합니다. 이는 ``libgomp`` 보다 더 나은 성능 이점을 제공할 때도 있습니다. 환경 변수 
# ``LD_PRELOAD`` 를 사용하여 OpenMP 라이브러리를 ``libiomp`` 로 전환할 수 있습니다.
#
# .. code-block:: sh
#
#    export LD_PRELOAD=<path>/libiomp5.so:$LD_PRELOAD

###############################################################################
# GNU OpenMP의 CPU 친화도 설정과 유사하게, ``libiomp`` 에서도 CPU 친화도 설정을 제어하는 환경 
# 변수가 제공됩니다. ``KMP_AFFINITY`` 는 OpenMP 스레드를 물리적 처리 장치에 바인딩합니다. 
# ``KMP_BLOCKTIME`` 은 스레드가 병렬 영역의 실행을 완료한 후 대기해야 하는 시간을 ms 단위로 
# 설정합니다. 대부분의 경우 ``KMP_BLOCKTIME`` 을 1 또는 0으로 설정하면 좋은 성능을 얻을 수 
# 있습니다. 다음 명령어는 Intel OpenMP 런타임 라이브러리에서의 일반적인 설정입니다.
#
# .. code-block:: sh
#
#    export KMP_AFFINITY=granularity=fine,compact,1,0
#    export KMP_BLOCKTIME=1

###############################################################################
# 메모리 할당자 전환
# ~~~~~~~~~~~~~~~~~~~~~~~
# 딥러닝 작업에서는 기존에 ``malloc`` 함수보다 메모리를 최대한 재사용할 수 있는 ``Jemalloc`` 
# 또는 ``TCMalloc`` 을 사용하면 더 나은 성능을 얻을 수 있습니다. 
# `Jemalloc <https://github.com/jemalloc/jemalloc>`_ 은 일반적인 목적의 ``malloc`` 
# 을 구현한 것으로, 단편화 방지와 확장 가능한 동시성 지원에 중점을 둡니다. 
# `TCMalloc <https://google.github.io/tcmalloc/overview.html>`_ 은 또한 프로그램 
# 실행 속도를 높이기 위한 몇 가지 최적화를 제공합니다. 그 중 하나는 메모리를 캐시에 보관하여 자주 
# 사용되는 객체에 대한 접근 속도를 높이는 것입니다. 이러한 캐시를 할당 해제 후에도 유지하면 나중에 
# 메모리가 다시 할당될 때 비용이 많이 드는 시스템 호출을 피하는 데 도움이 됩니다. 이들 중 하나를 
# 사용하려면 환경 변수 ``LD_PRELOAD`` 를 설정하십시오.
#
# .. code-block:: sh
#
#    export LD_PRELOAD=<jemalloc.so/tcmalloc.so>:$LD_PRELOAD

###############################################################################
# TorchScript로 추론 시 oneDNN Graph 사용하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# oneDNN Graph는 추론 성능을 크게 향상시킬 수 있습니다. 이는 합성곱, 행렬 곱셈(matmul)과 같은 
# 연산을 주변 연산과 결합하여 처리합니다. PyTorch 2.0에서는 ``Float32`` 및 ``BFloat16`` 
# 데이터 유형에 대해 베타 기능으로 지원됩니다. oneDNN Graph는 모델의 그래프를 받아서 예제 입력의 
# 모양을 고려하여 연산자 결합 후보를 식별합니다. 모델은 예제 입력을 사용하여 JIT-traced 되어야 
# 합니다. 동일한 모양의 입력에 대해 몇 번의 워밍업 후 속도가 향상될 수 있습니다. 아래의 예제 
# 코드는 resnet50에 대한 것이지만 사용자가 원하는 모델에서도 oneDNN Graph를 사용할 수 있습니다.

# oneDNN Graph를 사용하려면 이 추가 코드 한 줄로 충분합니다.
torch.jit.enable_onednn_fusion(True)

###############################################################################
# oneDNN Graph API를 사용하려면 Float32 추론 시 단 한 줄의 코드를 추가해야 합니다. oneDNN 
# Graph를 사용 중이라면 ``torch.jit.optimize_for_inference`` 호출을 피해야 합니다.

# sample_input은 예상되는 입력과 동일한 모양이어야 합니다.
sample_input = [torch.rand(32, 3, 224, 224)]
# resnet50 모델을 예시로 사용하지만, 아래 줄은 사용자가 원하는 모델에 맞게 수정할 수 있습니다.
model = getattr(torchvision.models, "resnet50")().eval()
# sample_input으로 모델을 trace하기
traced_model = torch.jit.trace(model, sample_input)
# torch.jit.freeze 호출하기
traced_model = torch.jit.freeze(traced_model)

###############################################################################
# 모델이 sample_input을 사용해 JIT-traced되면 몇 번의 워밍업 후 추론에 사용할 수 있습니다.

with torch.no_grad():
    # 몇 번의 워밍업
    traced_model(*sample_input)
    traced_model(*sample_input)
    # 워밍업 실행 후 성능 향상 관찰 가능
    traced_model(*sample_input)

###############################################################################
# oneDNN Graph용 JIT fuser는 ``BFloat16`` 데이터 타입을 사용한 추론도 지원하지만, oneDNN 
# Graph의 성능 이점은 AVX512_BF16 명령어 세트 아키텍처(ISA)의 머신에서 나타납니다. 
# 다음 코드 예시는 oneDNN Graph를 사용해 ``BFloat16`` 데이터 타입으로 추론하는 예시입니다.

# JIT 모드의 AMP는 기본적으로 활성화되어 있으며 eager 모드와 다르게 작동합니다.
torch._C._jit_set_autocast_mode(False)

with torch.no_grad(), torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
    # CNN 기반 비전 모델의 Conv-BatchNorm folding은 AMP를 사용할 때 ``torch.fx.experimental.optimization.fuse``를 통해 동작해야 합니다.
    import torch.fx.experimental.optimization as optimization
    # AMP를 사용하지 않는 경우 optimization.fuse를 호출할 필요는 없습니다.
    model = optimization.fuse(model)
    model = torch.jit.trace(model, (example_input))
    model = torch.jit.freeze(model)
    # 몇 번의 워밍업
    model(example_input)
    model(example_input)
    # 워밍업 실행 후 성능 향상 관찰 가능
    model(example_input)


###############################################################################
# PyTorch ``DistributedDataParallel`` (DDP) 기능을 사용해 CPU에서 모델 학습하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DLRM과 같은 소규모 모델 또는 메모리에 바인딩 된 모델의 경우 CPU에서 학습하는 것도 좋은 선택입니다. 
# 여러 소켓을 가진 머신에서는, 분산 학습으로 고효율의 하드웨어 자원을 사용하여 학습 과정을 가속할 수 있습니다. 
# `Torch-ccl <https://github.com/intel/torch-ccl>`_ 은 Intel(R)의 ``oneCCL`` 
# (집합 통신 라이브러리)로 최적화되어 효율적인 분산 딥러닝 학습을 위해 ``allreduce`` , 
# ``allgather`` , ``alltoall`` 과 같은 집합 연산을 구현합니다. Torch-ccl은 PyTorch C10D 
# ``ProcessGroup`` API를 구현하며, 외부 ``ProcessGroup`` 으로 동적으로 로드할 수 있습니다. 
# PyTorch DDP 모듈에서 구현된 최적화를 통해 ``torch-ccl`` 은 통신 연산을 가속화합니다. 통신 
# 커널의 최적화 외에도 ``torch-ccl`` 은 계산과 통신을 동시에 수행하는 기능을 제공합니다.

###############################################################################
# GPU 전용 최적화 방법
# --------------------------

###############################################################################
# cuDNN auto-tuner 활성화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# `NVIDIA cuDNN <https://developer.nvidia.com/cudnn>`_ 은 합성곱을 계산하기 위해 여러 
# 알고리즘을 지원합니다. Autotuner는 짧은 벤치마크를 실행하고 주어진 하드웨어와 입력 크기에 대해 
# 최상의 성능을 가진 커널을 선택합니다.
#
# 합성곱 신경망 (다른 유형은 현재 지원되지 않음)의 경우 학습하기 전에 cuDNN autotuner를 
# 활성화하려면 다음과 같이 설정하십시오:

torch.backends.cudnn.benchmark = True
###############################################################################
#
# * autotuner의 결정은 비결정적일 수 있습니다. 서로 다른 실행에서 다른 알고리즘이 선택될 수 
#   있습니다. 자세한 내용은 
#   `PyTorch: Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html?highlight=determinism>`_
#   를 참조하세요.
# * 드문 상황에서, 예를 들면 입력 크기가 가변적인 경우, 각 입력 크기에 대해 알고리즘 선택과 관련된 
#   오버헤드를 피하기 위해 autotuner를 비활성화하고 합성곱 신경망을 실행하는 것이 더 나을 수 있습니다.
#

###############################################################################
# 불필요한 CPU-GPU 동기화 피하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPU가 GPU 같은 가속기보다 최대한 앞서 실행될 수 있도록 불필요한 동기화를 피하여 가속기의 작업 큐에 
# 많은 작업이 포함되도록 하십시오.
#
# 가능하면 동기화를 요구하는 작업을 피하십시오. 예시:
#
# * ``print(cuda_tensor)``
# * ``cuda_tensor.item()``
# * 메모리 복사 : ``tensor.cuda()``,  ``cuda_tensor.cpu()`` 혹은 이에 상응하는
#   ``tensor.to(device)`` 호출
# * ``cuda_tensor.nonzero()``
# * CUDA tensor에서 수행된 연산 결과에 의존하는 파이썬 제어 흐름
#   예시: ``if (cuda_tensor != 0).all()``
#

###############################################################################
# 대상 장치에서 직접 tensor 생성하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``torch.rand(size).cuda()`` 를 호출하여 무작위 tensor를 생성하는 대신에 tensor를 직접 
# 장치에서 생성합니다.
# ``torch.rand(size, device='cuda')``
#
# 이는 다음과 같이 ``device`` 인수를 받아 새로운 tensor를 생성하는 모든 함수에 적용됩니다:
# `torch.rand() <https://pytorch.org/docs/stable/generated/torch.rand.html#torch.rand>`_,
# `torch.zeros() <https://pytorch.org/docs/stable/generated/torch.zeros.html#torch.zeros>`_,
# `torch.full() <https://pytorch.org/docs/stable/generated/torch.full.html#torch.full>`_
# 등.

###############################################################################
# 혼합 정밀도와 AMP 사용하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 혼합 정밀도는 
# `Tensor Cores <https://www.nvidia.com/en-us/data-center/tensor-cores/>`_
# 를 활용하여 Volta나 최신 GPU 아키텍처에서 최대 3배의 전체 속도 향상을 제공합니다. Tensor 
# Cores를 사용하려면 AMP를 활성화하고 행렬/텐서 차원이 Tensor Cores를 사용하는 커널 호출 
# 요구 사항을 충족해야 합니다.
#
# Tensor Cores를 사용하려면:
#
# * size를 8의 배수로 설정 (Tensor Cores의 차원에 맞추기 위해)
#
#   * `Deep Learning Performance Documentation
#     <https://docs.nvidia.com/deeplearning/performance/index.html#optimizing-performance>`_
#     에서 자세한 정보와 레이어 유형에 따른 가이드라인을 참조하세요.
#   * 레이어 크기가 고정되지 않고 다른 매개변수에서 유도되는 경우에도 명시적으로 패딩할 수 있습니다. 
#     (예시: NLP 모델의 어휘 크기 등).
#
# * AMP 활성화하기
#
#   * 혼합 정밀도 학습과 AMP 소개:
#     `video <https://www.youtube.com/watch?v=jF4-_ZK_tyc&feature=youtu.be>`_,
#     `slides <https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf>`_
#   * PyTorch 1.6부터 사용할 수 있는 PyTorch AMP:
#     `documentation <https://pytorch.org/docs/stable/amp.html>`_,
#     `examples <https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples>`_,
#     `tutorial <https://tutorials.pytorch.kr/recipes/recipes/amp_recipe.html>`_
#
#

###############################################################################
# 가변 입력 길이에 대비하여 메모리 미리 할당하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 음성 인식 또는 NLP 모델은 종종 가변 시퀀스 길이를 가진 tensor를 입력으로 학습됩니다. 가변 길이는 
# PyTorch 캐싱 할당기에서 문제를 일으킬 수 있으며, 성능 저하 또는 예기치 않은 메모리 부족 오류를 
# 초래할 수 있습니다. 짧은 시퀀스 길이의 배치가 더 긴 시퀀스 길이의 배치로 이어지면, PyTorch는 
# 이전 반복의 중간 버퍼를 해제하고 새 버퍼를 재할당해야 합니다. 이 과정은 시간이 많이 소요되며 캐싱 
# 할당기에서 조각화(fragmentation)를 일으켜 메모리 부족 오류를 유발할 수 있습니다.
#
# 일반적인 해결 방법은 미리 할당(preallocation)을 구현하는 것입니다. 
# 다음 단계로 구성됩니다:
#
# #. 최대 시퀀스 길이(훈련 데이터 세트의 최대 길이 또는 사전 정의된 임계값에 해당)를 갖는 (일반적으로 
#    무작위) 입력 배치를 생성합니다.
# #. 생성된 배치로 순방향 및 역방향 과정을 실행합니다. 옵티마이저나 학습률 스케줄러는 실행하지 않으며, 
#    이 단계는 이후 학습에서 재사용할 수 있는 최대 크기의 버퍼를 미리 할당합니다.
# #. 변화도를 0으로 설정합니다.
# #. 정규 학습을 진행합니다.
#

###############################################################################
# 분산 최적화 방법
# -------------------------

###############################################################################
# 효율적인 데이터 병렬 백엔드 사용하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch에는 데이터 병렬 학습을 구현하는 두 가지 방법이 있습니다:
#
# * `torch.nn.DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel>`_
# * `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
#
# ``DistributedDataParallel`` 은 다중 GPU에 대해 훨씬 더 나은 성능과 확장성을 제공합니다.
# 자세한 정보는 PyTorch 문서의
# `relevant section of CUDA Best Practices <https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel>`_
# 를 참조하세요.

###############################################################################
# 학습할 때 ``DistributedDataParallel`` 이나 변화도 축적 사용 시 불필요한 all-reduce 건너뛰는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 기본적으로
# `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
# 은 모든 역전파 과정 후에 변화도 all-reduce를 실행하여 학습에 참여하는 모든 워커에서의 평균 변화도를 
# 계산합니다. 학습 시 변화도 축적을 N단계 동안 사용하는 경우, 모든 학습 단계 후에 all-reduce가 
# 요하지 않습니다. 마지막 역전파 호출 직후, 즉 옵티마이저 실행 직전에만 all-reduce를 
# 수행하면 됩니다.
#
# ``DistributedDataParallel`` 은 특정 반복에 대해 변화도 all-reduce를 비활성화하는
# `no_sync() <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync>`_
# 컨텍스트 관리자를 제공합니다.
# ``no_sync()`` 는 변화도 축적의 첫 ``N-1`` 반복에 적용되어야 하며 마지막 반복은 기본 실행을 
# 따르고 필요한 변화도 all-reduce를 수행해야 합니다.

###############################################################################
# ``DistributedDataParallel(find_unused_parameters=True)`` 를 사용할 때 생성자와 실행 레이어 순서를 일치시키는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
# 은 ``find_unused_parameters=True`` 와 함께 모델 생성자에서의 레이어와 파라미터 순서를 
# 사용하여 ``DistributedDataParallel`` 변화도 all-reduce를 위한 버킷을 만듭니다. 
# ``DistributedDataParallel`` 은 all-reduce를 역전파와 겹치게 수행합니다. 특정 버킷에 대한 
# all-reduce는 주어진 버킷의 모든 파라미터에 대한 변화도가 모두 준비되었을 때 비동기적으로 작동됩니다.
#
# 최대로 겹치게 하려면 모델 생성자에서의 순서가 실제 실행 중인 순서와 대략적으로 일치해야 합니다. 
# 순서가 맞지 않으면 전체 버킷에 대한 all-reduce는 마지막으로 도착하는 변화도를 기다리게 되며, 
# 이는 역전파와 all-reduce 간의 겹침을 줄일 수 있고, all-reduce가 노출되어 학습 속도가 느려질 수 
# 있습니다.
#
# ``find_unused_parameters=False`` 가 (기본 설정)인 ``DistributedDataParallel`` 은 
# 역전파 중에 발견된 연산 순서를 기반으로 자동으로 버킷을 형성합니다. 
# ``find_unused_parameters=False`` 를 사용할 때는 최적의 성능을 달성하기 위해 레이어나 
# 파라미터의 순서를 재조정할 필요가 없습니다.

###############################################################################
# 분산 설정에서 작업 부하를 분산하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 작업 부하 불균형은 일반적으로 순차적인 데이터를 처리하는 모델(예시: 음성 인식, 번역, 언어 모델 등)
# 에서 발생할 수 있습니다. 하나의 장치가 나머지 장치들보다 긴 시퀀스 길이를 가진 데이터 배치를 받으면,
# 모든 장치가 마지막으로 작업을 끝내는 워커를 기다리게 됩니다. 역전파는 
# `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
# 백엔드와 함께 분산 설정에서 암묵적인 동기화 지점으로 작용합니다.
#
# 작업 로드 밸런싱 문제를 해결하는 방법은 여러 가지가 있습니다. 핵심은 각 전역 배치 내에서 
# 모든 워커에 걸쳐 작업 부하를 가능한 한 균일하게 분배하는 것입니다. 예를 들어, Transformer는 배치 
# 내에서 대략 일정한 수의 토큰(변동하는 수의 시퀀스)을 형성하여 불균형을 해결하며, 다른 모델은 유사한 
# 시퀀스 길이를 가진 샘플을 버킷화하거나 데이터셋을 시퀀스 길이에 따라 정렬하여 불균형을 해결합니다.
