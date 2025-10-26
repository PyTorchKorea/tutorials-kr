# -*- coding: utf-8 -*-
"""
`Introduction to ONNX <intro_onnx.html>`_ ||
**PyTorch 모델을 ONNX로 내보내기** ||
`Extending the ONNX exporter operator support <onnx_registry_tutorial.html>`_ ||
`Export a model with control flow to ONNX <export_control_flow_model_to_onnx_tutorial.html>`_

PyTorch 모델을 ONNX로 내보내기
==============================

**저자**: `Ti-Tai Wang <https://github.com/titaiwangms>`_, `Justin Chu <justinchu@microsoft.com>`_, `Thiago Crepaldi <https://github.com/thiagocrepaldi>`_.

**번역**: `이준혁 <https://github.com/titaiwangms>`_.

.. note::
    PyTorch 2.5부터 두 가지 ONNX 익스포터(Exporter) 옵션을 사용할 수 있습니다.
    * ``torch.onnx.export(..., dynamo=True)`` 는 그래프 생성에 ``torch.export`` 와 Torch FX를 활용하는 권장 익스포터입니다.
    * ``torch.onnx.export`` 는 TorchScript에 의존하는 레거시 방식이며 더 이상 사용이 권장되지 않습니다.
    
"""

###############################################################################
# `PyTorch로 딥러닝하기: 60분만에 끝장내기 <https://tutorials.pytorch.kr/beginner/deep_learning_60min_blitz.html>`_ 에서는
# PyTorch를 고수준에서 배우고 작은 이미지 분류 신경망을 학습시켜볼 수 있었습니다.
# 이 튜토리얼에서는 그 내용의 확장으로 ``torch.onnx.export(..., dynamo=True)`` ONNX 익스포터를 사용하여
# PyTorch에서 정의된 모델을 ONNX 형식으로 변환하는 방법을 알아보겠습니다.
#
# 모델 개발과 실험에는 PyTorch가 매우 유용하고, PyTorch로 완성된 모델은 `ONNX <https://onnx.ai/>`_ (Open Neural Network Exchange)
# 등 다양한 형식으로 변환해 실제 서비스 환경에 배포할 수 있습니다.
#
# ONNX는 머신러닝 모델을 나타내는 유연한 공개 표준 형식입니다. 이런 표준화된 표현을 사용하는 모델은
# 대규모 클라우드 기반 슈퍼컴퓨터부터, 웹 브라우저나 휴대폰처럼 리소스가 제한된 엣지 디바이스까지
# 다양한 하드웨어 플랫폼 및 런타임 환경에서 실행될 수 있습니다.
#
# 이 튜토리얼에서는 다음 항목들에 대해서 알아보겠습니다.
#
# 1. 필요한 의존성 설치하기.
# 2. 간단한 이미지 분류 모델 작성하기.
# 3. 모델을 ONNX 형식으로 내보내기.
# 4. ONNX 모델을 파일에 저장하기.
# 5. `Netron <https://github.com/lutzroeder/netron>`_ 을 사용해 ONNX 모델 그래프 시각화하기.
# 6. `ONNX Runtime` 으로 ONNX 모델 실행하기.
# 7. PyTorch의 결과와 ONNX Runtime의 결과 비교하기.
#
# 1. 필요한 의존성 설치하기
# ------------------------------------
# ONNX 익스포터는 PyTorch 연산자를 ONNX 연산자로 변환할 때 ``onnx`` 와 ``onnxscript`` 를 사용하므로 
# 이에 대한 설치를 진행합니다.
#
#  .. code-block:: bash
#
#   pip install --upgrade onnx onnxscript
#
# 2. 간단한 이미지 분류 모델 작성하기
# -----------------------------------------
#
# 환경 설정이 완료되었으면 `PyTorch로 딥러닝하기: 60분만에 끝장내기 <https://tutorials.pytorch.kr/beginner/deep_learning_60min_blitz.html>`_ 에서
# 했던 것처럼 PyTorch로 간단한 이미지 분류 모델을 작성합니다.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


######################################################################
# 3. 모델을 ONNX 형식으로 내보내기
# ----------------------------------
#
# 모델이 정의되었으므로, 이제 모델을 인스턴스화하고 임의의 32x32 입력을 생성합니다.
# 이후 모델을 ONNX 형식으로 내보냅니다.

torch_model = ImageClassifierModel()
# 모델을 내보내기 위한 예시 입력을 생성합니다. 이때 입력은 tensor의 튜플이어야 합니다.
example_inputs = (torch.randn(1, 1, 32, 32),)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)

######################################################################
# 지금까지 진행한 과정에서는 모델에 대한 어떠한 코드 변경도 필요하지 않았습니다.
# 결과로 도출된 ONNX 모델은 ``torch.onnx.ONNXProgram`` 내에 이진 protobuf 파일로 저장됩니다.
#
# 4. ONNX 모델을 파일에 저장하기
# --------------------------------
#
# 내보낸 모델을 메모리에 올려두고 유용하게 활용할 수도 있지만,
# 다음 코드로 모델을 디스크에 저장할 수도 있습니다.

onnx_program.save("image_classifier_model.onnx")

######################################################################
# 다음 코드로 ONNX 파일을 다시 메모리에 올리고 그 형식이 올바른지 확인할 수 있습니다.

import onnx

onnx_model = onnx.load("image_classifier_model.onnx")
onnx.checker.check_model(onnx_model)

######################################################################
# 5. Netron을 사용해 ONNX 모델 그래프 시각화하기
# ----------------------------------------------
#
# 모델이 파일에 저장되어 있으면 `Netron <https://github.com/lutzroeder/netron>`_ 으로 시각화할 수 있습니다.
# Netron은 macos, Linux 또는 Windows 컴퓨터에 설치하거나 웹 브라우저에서 직접 실행할 수 있습니다.
# 이 `링크 <https://github.com/lutzroeder/netron>`_ 로 웹 브라우저 버전을 사용해 보겠습니다.
#
# .. image:: ../../_static/img/onnx/netron_web_ui.png
#   :width: 70%
#   :align: center
#
#
# Netron이 열리면 ``image_classifier_model.onnx`` 파일을 브라우저로 드래그 앤드 드롭(drag and drop)하거나
# **Open model** 버튼을 클릭한 후 파일을 선택합니다.
#
# .. image:: ../../_static/img/onnx/image_classifier_onnx_model_on_netron_web_ui.png
#   :width: 50%
#
#
# 이제 됐습니다! 성공적으로 PyTorch 모델을 ONNX 형식으로 내보내고 Netron으로 시각화했습니다.
#
# 6. ONNX Runtime으로 ONNX 모델 실행하기
# -------------------------------------------
#
# 마지막 단계는 `ONNX Runtime` 으로 ONNX 모델을 실행하는 것입니다. 그 전에 먼저 ONNX Runtime을 설치하겠습니다.
#
#  .. code-block:: bash
#
#   pip install onnxruntime
#
# ONNX 표준은 PyTorch가 지원하는 모든 데이터 구조와 타입을 지원하지는 않으므로,
# ONNX Runtime에 넣기 전에 우선 PyTorch 입력을 ONNX 형식에 맞게 조정해야 합니다.
# 이 예제에서는 입력이 동일하지만, 더 복잡한 모델에서는
# 기존 PyTorch 모델보다 더 많은 입력으로 나누어야 할 수 있습니다.
#
# ONNX Runtime은 모든 PyTorch tensor를 (CPU의) Numpy tensor로 변환한 뒤,
# 입력 이름 문자열을 키로, Numpy tensor를 값으로 하는 딕셔너리로 감싸는 추가 단계가 필요합니다.
#
# 이제 *ONNX Runtime 추론 세션* 을 생성하고, 처리된 입력으로 ONNX 모델을 실행하여 출력을 얻을 수 있습니다.
# 이 튜토리얼에서 ONNX Runtime은 CPU에서 실행되지만, GPU에서도 실행될 수 있습니다.

import onnxruntime

onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
print(f"Input length: {len(onnx_inputs)}")
print(f"Sample input: {onnx_inputs}")

ort_session = onnxruntime.InferenceSession(
    "./image_classifier_model.onnx", providers=["CPUExecutionProvider"]
)

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime은 출력에 대한 리스트를 반환합니다.
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

####################################################################
# 7. PyTorch의 결과와 ONNX Runtime의 결과 비교하기
# ------------------------------------------------------------------
#
# 내보낸 모델이 올바르게 작동하는지 확인하는 가장 좋은 방법은
# 신뢰할 수 있는 기준(source of truth)이 되는 PyTorch와 수치적으로 비교하는 것입니다.
#
# 이를 위해 동일한 입력으로 PyTorch 모델을 실행하고 그 결과를 ONNX Runtime의 결과와 비교할 수 있습니다.
# 결과를 비교하기 전에 PyTorch의 출력을 ONNX의 형식과 일치하도록 변환해야 합니다.

torch_outputs = torch_model(*example_inputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")

######################################################################
# 결론
# ----------
#
# 이것으로 튜토리얼을 마칩니다! 우리는 성공적으로 PyTorch 모델을 ONNX 형식으로 내보내고,
# 모델을 디스크에 저장하고, Netron을 사용하여 시각화하고, ONNX Runtime으로 실행했으며,
# 마지막에는 ONNX Runtime의 결과를 PyTorch의 결과와 수치적으로 비교했습니다.
#
# 더 읽어보기
# ---------------
#
# 아래 목록은 기본 예제부터 고급 시나리오까지 아우르는 튜토리얼들로,
# 반드시 나열된 순서대로 볼 필요는 없습니다.
# 자유롭게 관심 가는 주제로 바로 이동하거나,
# 편안히 앉아 전체 튜토리얼을 하나씩 살펴보며 ONNX 익스포터에 대한 모든 것들을 배워보세요.
#
# .. include:: /beginner_source/onnx/onnx_toc.txt
#
# .. toctree::
#    :hidden:
#