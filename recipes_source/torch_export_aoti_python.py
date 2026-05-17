# -*- coding: utf-8 -*-

"""
.. meta::
   :description: Python 런타임에서 AOTInductor를 사용하는 엔드 투 엔드 예제
   :keywords: torch.export, AOTInductor, torch._inductor.aoti_compile_and_package, aot_compile, torch._export.aoti_load_package

(Beta) Python 런타임을 위한 ``torch.export`` AOTInductor 튜토리얼
==================================================================
**저자:** Ankith Gunapal, Bin Bao, Angela Yi
**번역:** `김정연 <https://github.com/jykimai>`_
"""

######################################################################
#
# .. warning::
#
#     ``torch._inductor.aoti_compile_and_package`` 와
#     ``torch._inductor.aoti_load_package`` 는 Beta 상태이며, 하위 호환성을 깨는
#     변경이 발생할 수 있습니다. 이 튜토리얼은 Python 런타임을 사용한
#     모델 배포에 이러한 API를 활용하는 방법을 예제로 보여줍니다.
#
# `이전 문서 <https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html#>`__ 에서
# AOTInductor를 사용하여 PyTorch로 내보낸(exported) 모델을 사전 컴파일(Ahead-of-Time compilation)하고,
# Python이 아닌 환경에서도 실행할 수 있는 산출물(artifact)을 생성하는 방법을 살펴보았습니다.
# 이 튜토리얼에서는 Python 런타임에서 AOTInductor를 사용하는 방법을 엔드 투 엔드 예제로 알아봅니다.
#
# **목차**
#
# .. contents::
#     :local:

######################################################################
# 전제조건
# -------------
# * PyTorch 2.6 이상
# * ``torch.export`` 와 AOTInductor에 대한 기본적인 이해
# * `AOTInductor: Torch.Export로 내보낸 모델의 사전 컴파일(Ahead-of-Time Compilation) <https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html#>`_ 튜토리얼 완료

######################################################################
# 이 튜토리얼에서 배울 내용
# ----------------------
# * Python 런타임에서 AOTInductor를 사용하는 방법
# * :func:`torch._inductor.aoti_compile_and_package` 와 :func:`torch.export.export` 를 함께 사용하여 컴파일된 산출물(artifact)을 생성하는 방법
# * :func:`torch._export.aot_load` 를 사용하여 Python 런타임에서 산출물을 불러오고 실행하는 방법
# * Python 런타임과 함께 AOTInductor를 사용해야 하는 경우

######################################################################
# 모델 컴파일
# -----------------
#
# 예시로 TorchVision의 사전 학습된 ``ResNet18`` 모델을 사용합니다.
#
# 첫 번째 단계는 :func:`torch.export.export` 를 사용하여 모델을 그래프 표현으로
# 내보내는 것입니다. 이 함수에 대해 더 자세히 알아보려면
# `문서 <https://pytorch.org/docs/main/export.html>`_ 나
# `튜토리얼 <https://tutorials.pytorch.kr/intermediate/torch_export_tutorial.html>`_ 을 참고하세요.
#
# PyTorch 모델을 내보내어 ``ExportedProgram`` 을 얻은 후에는,
# :func:`torch._inductor.aoti_compile_and_package` 를 AOTInductor에 적용하여
# 지정된 디바이스에 맞춰 프로그램을 컴파일하고, 생성된 내용을 ".pt2" 산출물로 저장할 수 있습니다.
#
# .. note::
#
#       이 API는 :func:`torch.compile` 에서 사용할 수 있는 옵션과 동일한 옵션을 지원합니다.
#       예를 들어 ``mode`` 와 ``max_autotune`` 같은 옵션이 있으며,
#       이는 CUDA 그래프를 활성화하고 Triton 기반의 행렬 곱셈과 합성곱(convolution)을
#       활용하고자 하는 경우에 사용합니다.

import os
import torch
import torch._inductor
from torchvision.models import ResNet18_Weights, resnet18

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

with torch.inference_mode():
    inductor_configs = {}

    if torch.cuda.is_available():
        device = "cuda"
        inductor_configs["max_autotune"] = True
    else:
        device = "cpu"

    model = model.to(device=device)
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)

    exported_program = torch.export.export(
        model,
        example_inputs,
    )
    path = torch._inductor.aoti_compile_and_package(
        exported_program,
        package_path=os.path.join(os.getcwd(), "resnet18.pt2"),
        inductor_configs=inductor_configs
    )

######################################################################
# :func:`aoti_compile_and_package` 의 결과로 "resnet18.pt2" 산출물이 생성되며,
# Python과 C++ 환경 모두에서 불러와 실행할 수 있습니다.
#
# 산출물 자체에는 AOTInductor가 생성한 다양한 코드가 포함되어 있습니다.
# 예를 들어 생성된 C++ 러너 파일, C++ 파일로부터 컴파일된 공유 라이브러리,
# 그리고 CUDA에 최적화하는 경우에는 CUDA 바이너리 파일(cubin 파일)이 함께 들어 있습니다.
#
# 구조 측면에서 산출물은 다음과 같은 명세를 가진 구조화된 ``.zip`` 파일입니다.
#
# .. code::
#    .
#    ├── archive_format
#    ├── version
#    ├── data
#    │   ├── aotinductor
#    │   │   └── model
#    │   │       ├── xxx.cpp            # AOTInductor가 생성한 cpp 파일
#    │   │       ├── xxx.so             # AOTInductor가 생성한 공유 라이브러리
#    │   │       ├── xxx.cubin          # Cubin 파일 (CUDA에서 실행하는 경우)
#    │   │       └── xxx_metadata.json  # 저장할 추가 메타데이터
#    │   ├── weights
#    │   │  └── TBD
#    │   └── constants
#    │      └── TBD
#    └── extra
#        └── metadata.json
#
# 다음 명령어를 사용하여 산출물의 내용을 확인할 수 있습니다.
#
# .. code:: bash
#
#    $ unzip -l resnet18.pt2
#
# .. code:: bash
#
#    $ unzip -l resnet18.pt2
#
# .. code::
#
#    Archive:  resnet18.pt2
#      Length      Date    Time    Name
#    ---------  ---------- -----   ----
#            1  01-08-2025 16:40   version
#            3  01-08-2025 16:40   archive_format
#        10088  01-08-2025 16:40   data/aotinductor/model/cagzt6akdaczvxwtbvqe34otfe5jlorktbqlojbzqjqvbfsjlge4.cubin
#        17160  01-08-2025 16:40   data/aotinductor/model/c6oytfjmt5w4c7onvtm6fray7clirxt7q5xjbwx3hdydclmwoujz.cubin
#        16616  01-08-2025 16:40   data/aotinductor/model/c7ydp7nocyz323hij4tmlf2kcedmwlyg6r57gaqzcsy3huneamu6.cubin
#        17776  01-08-2025 16:40   data/aotinductor/model/cyqdf46ordevqhiddvpdpp3uzwatfbzdpl3auj2nx23uxvplnne2.cubin
#        10856  01-08-2025 16:40   data/aotinductor/model/cpzfebfgrusqslui7fxsuoo4tvwulmrxirc5tmrpa4mvrbdno7kn.cubin
#        14608  01-08-2025 16:40   data/aotinductor/model/c5ukeoz5wmaszd7vczdz2qhtt6n7tdbl3b6wuy4rb2se24fjwfoy.cubin
#        11376  01-08-2025 16:40   data/aotinductor/model/csu3nstcp56tsjfycygaqsewpu64l5s6zavvz7537cm4s4cv2k3r.cubin
#        10984  01-08-2025 16:40   data/aotinductor/model/cp76lez4glmgq7gedf2u25zvvv6rksv5lav4q22dibd2zicbgwj3.cubin
#        14736  01-08-2025 16:40   data/aotinductor/model/c2bb5p6tnwz4elgujqelsrp3unvkgsyiv7xqxmpvuxcm4jfl7pc2.cubin
#        11376  01-08-2025 16:40   data/aotinductor/model/c6eopmb2b4ngodwsayae4r5q6ni3jlfogfbdk3ypg56tgpzhubfy.cubin
#        11624  01-08-2025 16:40   data/aotinductor/model/chmwe6lvoekzfowdbiizitm3haiiuad5kdm6sd2m6mv6dkn2zk32.cubin
#        15632  01-08-2025 16:40   data/aotinductor/model/c3jop5g344hj3ztsu4qm6ibxyaaerlhkzh2e6emak23rxfje6jam.cubin
#        25472  01-08-2025 16:40   data/aotinductor/model/chaiixybeiuuitm2nmqnxzijzwgnn2n7uuss4qmsupgblfh3h5hk.cubin
#       139389  01-08-2025 16:40   data/aotinductor/model/cvk6qzuybruhwxtfblzxiov3rlrziv5fkqc4mdhbmantfu3lmd6t.cpp
#           27  01-08-2025 16:40   data/aotinductor/model/cvk6qzuybruhwxtfblzxiov3rlrziv5fkqc4mdhbmantfu3lmd6t_metadata.json
#     47195424  01-08-2025 16:40   data/aotinductor/model/cvk6qzuybruhwxtfblzxiov3rlrziv5fkqc4mdhbmantfu3lmd6t.so
#    ---------                     -------
#     47523148                     18 files


######################################################################
# Python에서의 모델 추론
# -------------------------
#
# Python에서 산출물을 불러와 실행하려면 :func:`torch._inductor.aoti_load_package` 를 사용할 수 있습니다.
#

import os
import torch
import torch._inductor

model_path = os.path.join(os.getcwd(), "resnet18.pt2")

compiled_model = torch._inductor.aoti_load_package(model_path)
example_inputs = (torch.randn(2, 3, 224, 224, device=device),)

with torch.inference_mode():
    output = compiled_model(example_inputs)


######################################################################
# Python 런타임과 함께 AOTInductor를 사용해야 하는 경우
# ---------------------------------------------
#
# Python 런타임과 함께 AOTInductor를 사용하는 주된 이유는 크게 두 가지입니다.
#
# -  ``torch._inductor.aoti_compile_and_package`` 는 하나의 직렬화된 산출물을
#    생성합니다. 이는 배포 시 모델 버전 관리와 시간에 따른 모델 성능 추적에 유용합니다.
# -  :func:`torch.compile` 은 JIT 컴파일러이므로 첫 컴파일 시 워밍업 비용이 발생합니다.
#    따라서 배포 시 첫 추론에 걸리는 컴파일 시간을 고려해야 합니다.
#    반면 AOTInductor를 사용하면 ``torch.export.export`` 와
#    ``torch._inductor.aoti_compile_and_package`` 를 통해 컴파일이 미리 수행됩니다.
#    배포 시점에는 모델을 불러온 후 추론을 실행할 때 추가 비용이 발생하지 않습니다.
#
#
# 아래 섹션에서는 AOTInductor를 사용했을 때 첫 추론에서 얻을 수 있는 속도 향상을 보여줍니다.
#
# 추론에 걸리는 시간을 측정하기 위해 ``timed`` 라는 유틸리티 함수를 정의합니다.
#

import time
def timed(fn):
    # `fn()` 을 실행한 결과와 `fn()` 의 실행 시간(초)을 반환합니다.
    # CUDA를 지원하는 디바이스에서 정확하게 측정하기 위해
    # CUDA 이벤트와 동기화를 사용합니다.
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.time()

    result = fn()
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
    else:
        end = time.time()

    # 함수 실행에 걸린 시간을 밀리초 단위로 측정합니다.
    if torch.cuda.is_available():
        duration = start.elapsed_time(end)
    else:
        duration = (end - start) * 1000

    return result, duration


######################################################################
# AOTInductor를 사용한 첫 추론 시간을 측정해 보겠습니다.

torch._dynamo.reset()

model = torch._inductor.aoti_load_package(model_path)
example_inputs = (torch.randn(1, 3, 224, 224, device=device),)

with torch.inference_mode():
    _, time_taken = timed(lambda: model(example_inputs))
    print(f"Time taken for first inference for AOTInductor is {time_taken:.2f} ms")


######################################################################
# ``torch.compile`` 을 사용한 첫 추론 시간을 측정해 보겠습니다.

torch._dynamo.reset()

model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
model.eval()

model = torch.compile(model)
example_inputs = torch.randn(1, 3, 224, 224, device=device)

with torch.inference_mode():
    _, time_taken = timed(lambda: model(example_inputs))
    print(f"Time taken for first inference for torch.compile is {time_taken:.2f} ms")

######################################################################
# AOTInductor를 사용하면 ``torch.compile`` 에 비해 첫 추론 시간이 크게 단축되는 것을 확인할 수 있습니다.

######################################################################
# 결론
# ----------
#
# 이 튜토리얼에서는 사전 학습된 ``ResNet18`` 모델을 컴파일하고 불러오는 방법을 통해
# Python 런타임에서 AOTInductor를 효과적으로 사용하는 방법을 알아보았습니다.
# 이 과정은 컴파일된 산출물을 생성하고 Python 환경에서 실행하는 실용적인 활용 방법을 보여줍니다.
# 또한 첫 추론 시간 단축이라는 측면에서 모델 배포에 AOTInductor를 사용했을 때의 장점도 함께 살펴보았습니다.
