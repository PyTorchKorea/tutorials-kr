"""
(beta) torch.compile과 함께 TORCH_LOGS 파이썬 API 사용하기
==========================================================================================
**저자:** `Michael Lazos <https://github.com/mlazos>`_
**번역:** `장효영 <https://github.com/hyoyoung>`_
"""

import logging

######################################################################
#
# This tutorial introduces the ``TORCH_LOGS`` environment variable, as well as the Python API, and
# demonstrates how to apply it to observe the phases  of ``torch.compile``.
# 이 튜토리얼에서는 ``TORCH_LOGS`` 환경 변수와 함께 Python API를 소개하고,
# 이를 적용하여 ``torch.compile``의 단계를 관찰하는 방법을 보여줍니다.
#
# .. note::
#
#   이 튜토리얼에는 PyTorch 2.2.0 이상 버전이 필요합니다.
#
#


######################################################################
# 설정
# ~~~~~~~~~~~~~~~~~~~~~
# In this example, we'll set up a simple Python function which performs an elementwise
# add and observe the compilation process with ``TORCH_LOGS`` Python API.
# 이 예제에서는 요소별 덧셈을 수행하는 간단한 파이썬 함수를 설정하고
# ``TORCH_LOGS`` 파이썬 API를 사용하여 컴파일 프로세스를 관찰해 보겠습니다.
#
# .. note::
#
#   명령줄에서 로깅 설정을 변경하는 데 사용할 수 있는
#   환경 변수 ``TORCH_LOGS``도 있습니다. 각 예제에 해당하는
#   환경 변수 설정이 표시되어 있습니다.

import torch

# torch.compile을 지원하지 않는 기기인 경우 완전히 종료합니다.
if torch.cuda.get_device_capability() < (7, 0):
    print("Skipping because torch.compile is not supported on this device.")
else:
    @torch.compile()
    def fn(x, y):
        z = x + y
        return z + 2


    inputs = (torch.ones(2, 2, device="cuda"), torch.zeros(2, 2, device="cuda"))


# 각 예제 사이의 구분 기호를 출력하고 dynamo를 reset합니다
    def separator(name):
        print(f"==================={name}=========================")
        torch._dynamo.reset()


    separator("Dynamo Tracing")
# dynamo tracing 보기
# TORCH_LOGS="+dynamo"
    torch._logging.set_logs(dynamo=logging.DEBUG)
    fn(*inputs)

    separator("Traced Graph")
# traced 그래프 보기
# TORCH_LOGS="graph"
    torch._logging.set_logs(graph=True)
    fn(*inputs)

    separator("Fusion Decisions")
# fusion decision 보기
# TORCH_LOGS="fusion"
    torch._logging.set_logs(fusion=True)
    fn(*inputs)

    separator("Output Code")
# inductor가 생성한 결과 코드 보기
# TORCH_LOGS="output_code"
    torch._logging.set_logs(output_code=True)
    fn(*inputs)

    separator("")

######################################################################
# 결론
# ~~~~~~~~~~
#
# 이 튜토리얼에서는 사용 가능한 몇 가지 로깅 옵션을 실험하여
# TORCH_LOGS 환경 변수와 Python API를 소개했습니다.
# 사용 가능한 모든 옵션에 대한 설명을 보려면
# 파이썬 스크립트에서 import torch를 실행하고 TORCH_LOGS를 "help"로 설정하세요.
#
# 다른 방법으로는, `torch._logging 문서`_ 를 보면,
# 사용 가능한 모든 로깅 옵션에 대한 설명을 확인할 수 있습니다.
#
# torch.compile에 관한 더 많은 정보는, `torch.compile 튜토리얼`_를 보세요.
#
# .. _torch._logging 문서: https://pytorch.org/docs/main/logging.html
# .. _torch.compile 튜토리얼: https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html
