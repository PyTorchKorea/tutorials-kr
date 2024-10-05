# -*- coding: utf-8 -*-
"""
자동 혼합 정밀도(Automatic Mixed Precision) 가이드
*********************************************
**저자**: `Michael Carilli <https://github.com/mcarilli>`_
**역자**: `오왕택 <https://github.com/ohkingtaek>`_

`torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_ 에서는 혼합 정밀도(mixed precision)를 위해 편리한 메소드를 제공합니다.
이 방식에서는 일부 연산이 ``torch.float32`` (``float``) 데이터 타입을 사용하고, 다른 연산은 ``torch.float16`` (``half``) 을 사용합니다. 
예를 들어, 선형 계층이나 합성곱 같은 연산은 ``float16`` 또는 ``bfloat16`` 에서 훨씬 빠르게 실행됩니다. 
반면, 합계(sum), 평균(mean), 최대/최소값 연산과 같은 reduction 연산은 값의 범위 변동이 크므로 더 넓은 동적 범위를 제공하는 ``float32`` 가 필요합니다. 
혼합 정밀도는 각 연산에 적합한 데이터 타입을 매칭하여, 네트워크의 실행 시간과 메모리 사용량을 줄이는 데 도움을 줍니다.

일반적으로 "자동 혼합 정밀도 학습"은 `torch.autocast <https://pytorch.org/docs/stable/amp.html#torch.autocast>`_ 와
`torch.cuda.amp.GradScaler <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`_ 를 함께 사용합니다.

이 가이드에서는 기본 정밀도에서 간단한 네트워크의 성능을 측정한 후, ``autocast`` 와 ``GradScaler`` 를 추가하여 동일한 신경망을 혼합 정밀도로 실행해 성능을 향상시키는 과정을 설명합니다.

이 가이드는 파이썬 스크립트로 다운로드하여 실행할 수 있습니다. 필요한 요구 사항은 PyTorch 1.6 이상과 CUDA를 지원하는 GPU입니다.

혼합 정밀도는 주로 Tensor Core가 지원되는 아키텍처(Volta, Turing, Ampere)에서 좋은 성능을 냅니다. 
이러한 아키텍처에서는 2~3배의 성능 향상이 나타날 수 있습니다. 
이전 아키텍처(Kepler, Maxwell, Pascal)에서는 약간의 성능 향상이 있을 수 있습니다. 
GPU의 아키텍처를 확인하려면 ``nvidia-smi`` 명령을 실행하세요.
"""

import torch, time, gc

# 시간 처리에 사용할 함수들
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

##########################################################
# 간단한 신경망
# -----------
# 다음과 같은 선형 계층과 ReLU 연산의 연속은 혼합 정밀도를 사용했을 때
# 성능 향상을 보여줄 수 있습니다.

def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()

##########################################################
# ``batch_size``, ``in_size``, ``out_size``, 그리고 ``num_layers`` 는
# GPU에 충분한 작업량을 부여하기 위해 크게 설정했습니다. 일반적으로 GPU가 충분히
# 사용될 때 혼합 정밀도가 가장 큰 성능 향상을 제공합니다. 작은 신경망은 CPU에 의해
# 제약을 받을 수 있으며, 이 경우 혼합 정밀도는 성능을 향상시키지 못할 수 있습니다.
# 또한, 선형 계층에서 사용되는 차원들은 8의 배수로 설정되어 Tensor Core를
# 지원하는 GPU에서 Tensor Core를 사용할 수 있게 구성되었습니다.
# (아래 :ref:`Troubleshooting 해결 <troubleshooting>` 참조)
#
# 연습 문제: 사용할 사이즈를 다양하게 설정하여 혼합 정밀도 사용 시 성능 향상이 어떻게 달라지는지 확인해 보세요.

batch_size = 512 # 128, 256, 513도 시도해보세요.
in_size = 4096
out_size = 4096
num_layers = 3
num_batches = 50
epochs = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# 기본 정밀도로 데이터를 생성합니다.
# 아래에서 기본 정밀도와 혼합 정밀도 실험 모두에서 동일한 데이터를 사용합니다.
# 혼합 정밀도를 활성화할 때 입력의 ``dtype`` 을 수동으로 변경할 필요는 없습니다.
data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size) for _ in range(num_batches)]

loss_fn = torch.nn.MSELoss().cuda()

##########################################################
# 기본 정밀도
# ---------
# ``torch.cuda.amp`` 없이, 아래의 간단한 신경망은 모든 연산을 기본
# 정밀도 (``torch.float32``) 로 실행합니다.

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad() # 여기서 set_to_none=True는 성능을 약간 향상시킬 수 있습니다.
end_timer_and_print("Default precision:")

##########################################################
# ``torch.autocast`` 추가하는 방법
# -----------------------------
# `torch.autocast <https://pytorch.org/docs/stable/amp.html#autocasting>`_
# 의 인스턴스는 스크립트의 일부 영역을 혼합 정밀도로 실행할 수 있도록, 컨텍스트 관리자로 작동합니다.
# 이 영역에서 CUDA 연산은 성능을 개선하면서 정확도를 유지하기 위해 ``autocast`` 가 선택한 ``dtype`` 으로 실행됩니다.
# 각 연산에 대해 ``autocast`` 가 선택하는 정밀도와 해당 정밀도를 선택하는 상황에 대한 자세한 내용은
# `Autocast Op Reference <https://pytorch.org/docs/stable/amp.html#autocast-op-reference>`_ 를 참조하세요.

for epoch in range(0): # 0 epoch으로 설정한 것은 이 섹션을 설명하기 위한 것입니다.
    for input, target in zip(data, targets):
        # ``autocast`` 아래에서 순전파를 실행합니다
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            # output은 float16입니다. 이는 선형 계층이 ``autocast`` 에 의해 float16으로 변환되기 때문입니다.
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # loss는 float32입니다. 이는 ``mse_loss`` 계층이 ``autocast`` 에 의해 float32로 변환되기 때문입니다.
            assert loss.dtype is torch.float32

        # ``autocast`` 에서 backward() 전에 종료합니다.
        # ``autocast`` 에서 역전파를 실행하는 것은 권장되지 않습니다.
        # 역전파 연산은 해당 순전파 연산을 위해 ``autocast`` 가 선택한 것과 동일한 ``dtype`` 으로 실행됩니다.
        loss.backward()
        opt.step()
        opt.zero_grad() # 여기서 set_to_none=True는 성능을 약간 향상시킬 수 있습니다.

##########################################################
# ``GradScaler`` 추가하는 방법
# -------------------------
# `Gradient scaling <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`_
# 은 혼합 정밀도로 학습할 때 작은 크기의 변화도가 0으로 사라지는 ("underflowing") 하는 것을 방지하는 데 도움을 줍니다.
# `torch.cuda.amp.GradScaler <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`_
# 는 변화도 스케일링 단계를 편리하게 수행합니다.

# 수렴 실행의 시작에서 기본 인자를 사용하여 한 번 ``scaler`` 를 생성합니다.
# 신경망이 기본 ``GradScaler`` 인수로 수렴하지 않는다면, 이슈를 제출해주세요.
# 수렴 실행 전체에서 동일한 ``GradScaler`` 인스턴스를 사용해야 합니다.
# 동일한 스크립트에서 여러 번의 수렴 실행을 수행할 경우, 각 실행은 전용의 새로운 ``GradScaler`` 인스턴스를 사용해야 합니다.
# ``GradScaler`` 인스턴스는 가볍습니다.
scaler = torch.cuda.amp.GradScaler()

for epoch in range(0): # 0 epoch으로 설정한 것은 이 섹션을 설명하기 위한 것입니다.
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)

        # 손실을 조정합니다. 조정된 손실에 대해 ``backward()`` 를 호출하여 조정된 변화도를 생성합니다.
        scaler.scale(loss).backward()

        # ``scaler.step()`` 은 먼저 옵티마이저에 할당된 매개변수의 변화도를 복원합니다.
        # 이 변화도에 ``inf`` 나 ``NaN`` 이 포함되어 있지 않으면 optimizer.step()이 호출됩니다.
        # 그렇지 않으면 optimizer.step()을 건너뜁니다.
        scaler.step(opt)

        # 다음 반복을 위해 조정된 값을 업데이트합니다.
        scaler.update()

        opt.zero_grad() # 여기서 set_to_none=True는 성능을 약간 향상시킬 수 있습니다.

##########################################################
# "Automatic Mixed Precision" 소개한 것들 같이 사용하는 방법
# ---------------------------------------------------
# (다음 예시는 ``autocast`` 와 ``GradScaler`` 을 사용할 수 있는 편리한
# 인자인 ``enabled`` 를 보여줍니다. 만약 False로 설정되면, ``autocast``
# 와 ``GradScaler`` 의 호출이 무효화됩니다. 이를 통해 if/else 문 없이
# 기본 정밀도와 혼합 정밀도 간 전환이 가능합니다.)

use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # 여기서 set_to_none=True는 성능을 약간 향상시킬 수 있습니다.
end_timer_and_print("Mixed precision:")

##########################################################
# 변화도 확인/수정하기 (예: 클리핑)
# --------------------------
# ``scaler.scale(loss).backward()`` 로 생성된 모든 변화도는 조정됩니다.
# 만약 ``backward()`` 와 ``scaler.step(optimizer)`` 사이에서 파라미터의
# ``.grad`` 속성을 수정하거나 확인하고 싶다면, 먼저
# `scaler.unscale_(optimizer) <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.unscale_>`_
# 를 사용하여 변화도를 복원해야 합니다.

for epoch in range(0): # 0 epoch으로 설정한 것은 이 섹션을 설명하기 위한 것입니다.
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # 옵티마이저에 할당된 파라미터의 변화도를 제자리에서 복원합니다.
        scaler.unscale_(opt)

        # 옵티마이저에 할당된 파라미터의 변화도가 이제 복원되었으므로, 평소와 같이 클리핑할 수 있습니다. 
        # 이때 클리핑에 사용하는 max_norm 값은 변화도 조정이 없을 때와 동일하게 사용할 수 있습니다.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # 여기서 set_to_none=True는 성능을 약간 향상시킬 수 있습니다.

##########################################################
# 저장/재개하는 법
# --------------
# Amp가 활성화 상태에서 비트 단위의 정확도로 저장/재개하려면,
# `scaler.state_dict <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.state_dict>`_ 나
# `scaler.load_state_dict <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.load_state_dict>`_
# 를 사용하세요.
#
# 저장할 때는, 일반적인 모델과 옵티마이저의 상태와 함께 ``scaler`` 의 상태도 저장해야 합니다.
# 이를 각 반복하는 시작 시점, 즉 어떤 순전파 전에 하거나, 반복이 끝난 후에 ``scaler.update()`` 이후에 수행하면 됩니다.

checkpoint = {"model": net.state_dict(),
              "optimizer": opt.state_dict(),
              "scaler": scaler.state_dict()}
# 예를 들어 체크포인트를 작성하려면,
# torch.save(checkpoint, "filename")

##########################################################
# 재개할 때는, 모델과 옵티마이저의 상태와 함께 ``scaler`` 의 상태도
# 로드합니다. 예를 들어 체크포인트를 읽으려면,
#
# .. code-block::
#
#    dev = torch.cuda.current_device()
#    checkpoint = torch.load("filename",
#                            map_location = lambda storage, loc: storage.cuda(dev))
#
net.load_state_dict(checkpoint["model"])
opt.load_state_dict(checkpoint["optimizer"])
scaler.load_state_dict(checkpoint["scaler"])

##########################################################
# 체크포인트가 Amp 없이 생성된 경우, Amp를 사용하여 훈련을 재개하고 싶다면, 
# 모델과 옵티마이저 상태를 평소처럼 체크포인트에서 로드합니다. 이 체크포인트에는 
# 저장된 ``scaler`` 상태가 없으므로 새로운 ``GradScaler`` 인스턴스를 사용해야 합니다.
#
# 반대로 체크포인트가 Amp로 생성된 경우, ``Amp`` 를 사용하지 않고 훈련을 재개하려면,
# 모델과 옵티마이저 상태를 체크포인트에서 평소처럼 로드하고, 저장된 ``scaler`` 상태는 무시하면 됩니다.

##########################################################
# 추론/평가
# --------
# ``autocast`` 는 단독으로 사용하여 추론 또는 평가의 순전파를 감쌀 수 있습니다. 이 경우 ``GradScaler`` 는 필요하지 않습니다.

##########################################################
# .. _advanced-topics:
#
# 고급 주제
# ---------------
# 고급 사용 사례에 대한 내용은 `Automatic Mixed Precision Examples <https://pytorch.org/docs/stable/notes/amp_examples.html>`_
# 를 참조하세요. 예시에는 다음과 같은 내용이 포함되어 있습니다:
#
# * 변화도 축적
# * 변화도 페널티/이중 역전파
# * 다중 모델, 옵티마이저 또는 손실을 사용하는 신경망
# * 다중 GPU (``torch.nn.DataParallel`` 또는 ``torch.nn.parallel.DistributedDataParallel``)
# * 사용자 정의 autograd 함수 (``torch.autograd.Function`` 의 서브클래스)
#
# 동일한 스크립트에서 여러 번의 수렴 실행을 수행하는 경우, 각 실행은 새로운 ``GradScaler``
# 인스턴스를 사용해야 합니다. ``GradScaler`` 인스턴스는 가볍습니다.
#
# 만약 사용자 정의 C++ 연산을 디스패처에 등록하려면, 디스패처 튜토리얼의 
# `autocast section <https://tutorials.pytorch.kr/advanced/dispatcher.html#autocast>`_
# 을 참조하세요.

##########################################################
# .. _troubleshooting:
#
# Troubleshooting 해결
# -------------------
# Amp를 사용한 속도 향상이 미미한 경우
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. 네트워크가 GPU(들)에 충분한 작업을 제공하지 못하고 있어, CPU에 의한 병목 현상이 발생할 수 있습니다. 
#    이 경우 Amp의 GPU 성능에 대한 효과는 중요하지 않을 수 있습니다.
#
#    * GPU를 포화 상태로 만들기 위한 대략적인 방법은, 메모리 부족(OOM)이 발생하지 않는 선에서 가능한 한 배치 크기나 네트워크 크기를 늘리는 것입니다.
#    * 과도한 CPU-GPU 동기화(예: ``.item()`` 호출이나 CUDA 텐서에서 값을 출력하는 것)를 피해야 합니다.
#    * 사소한데 많은 CUDA 연산들을 피하고, 가능한 한 이러한 연산들을 몇 개의 큰 CUDA 연산으로 합치는 것이 좋습니다.
# 2. 신경망이 GPU 계산 병목 현상을 겪고 있을 수 있지만 (많은 ``matmul`` 또는 합성곱 연산), 사용 중인 GPU에 텐서 코어가 없을 수 있습니다. 
#    이 경우 속도 향상이 적을 수 있습니다.
# 3. ``matmul`` 차원이 Tensor Core에 친화적이지 않을 수 있습니다. ``matmul`` 에 참여하는 사이즈가 8의 배수인지 확인해야 합니다.
#    (인코더/디코더가 있는 NLP 모델의 경우, 이 점이 미묘할 수 있습니다. 또한, 합성곱 연산도 Tensor Core 사용을 위해 유사한 크기 제약을 가졌었지만, 
#    CuDNN 7.3 버전 이후로는 이러한 제약이 없습니다. 자세한 내용은 `여기 <https://github.com/NVIDIA/apex/issues/221#issuecomment-478084841>`_
#    에서 확인할 수 있습니다.)
#
# 손실이 inf/NaN인 경우
# ~~~~~~~~~~~~~~~~~~
# 먼저 신경망이 :ref:`고급 사용 사례 <advanced-topics>` 에 해당하는지 확인하세요.
# 또한 `Prefer binary_cross_entropy_with_logits over binary_cross_entropy <https://pytorch.org/docs/stable/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy>`_
# 을 참조하세요.
#
# Amp 사용이 올바르다고 확신한다면, 이슈를 제기해야 할 수도 있지만, 그 전에 다음 정보를 수집하는 것이 도움이 될 수 있습니다:
#
# 1. ``autocast`` 또는 ``GradScaler`` 를 각각 비활성화 (``enabled=False`` 를 생성자에 전달)하고 ``inf`` 나 ``NaN`` 이 여전히 발생하는지 확인합니다.
# 2. 신경망의 일부(예: 복잡한 손실 함수)가 오버플로우되는 것이 의심된다면, 해당 순전파 영역을 ``float32`` 로 실행하고 ``inf`` 나 ``NaN`` 이 발생하는지 확인하세요. 
#    `The autocast docstring <https://pytorch.org/docs/stable/amp.html#torch.autocast>`_ 
#    의 마지막 코드 단락에서 ``autocast`` 를 로컬로 비활성화하고 하위 영역의 입력을 캐스팅하여 하위 영역을 ``float32`` 로 실행하는 방법을 확인할 수 있습니다.
#
# 타입 불일치 오류 (``CUDNN_STATUS_BAD_PARAM`` 으로 나타날 수 있음)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``Autocast`` 는 캐스팅으로 이득을 얻거나 필요한 연산들을 모두 처리하려고 합니다.
# `명시적으로 다루어진 연산들 <https://pytorch.org/docs/stable/amp.html#autocast-op-reference>`_
# 은 수치적 특성뿐만 아니라 경험에 기반하여 선택되었습니다.
# ``Autocast`` 가 활성화된 순전파 영역이나 그 영역을 따른 역전파에서 타입 불일치 오류가 발생한다면, ``autocast`` 가 어떤 연산을 놓쳤을 가능성이 있습니다.
# 오류 추적내용과 함께 이슈를 제기하세요. 세부 정보 제공을 위해 스크립트를 실행하기 전에 ``export TORCH_SHOW_CPP_STACKTRACES=1``
# 을 설정하여 어느 백엔드 연산에서 실패하는지 확인할 수 있습니다.
