"""
`파이토치(PyTorch) 기본 익히기 <intro.html>`_ ||
`빠른 시작 <quickstart_tutorial.html>`_ ||
`텐서(Tensor) <tensorqs_tutorial.html>`_ ||
`Dataset과 Dataloader <data_tutorial.html>`_ ||
`변형(Transform) <transforms_tutorial.html>`_ ||
`신경망 모델 구성하기 <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`최적화(Optimization) <optimization_tutorial.html>`_ ||
**모델 저장하고 불러오기**

모델 저장하고 불러오기
==========================================================================

이번 장에서는 저장하기나 불러오기를 통해 모델의 상태를 유지(persist)하고 모델의 예측을 실행하는 방법을 알아보겠습니다.
"""

import torch
import torchvision.models as models


#######################################################################
# 모델 가중치 저장하고 불러오기
# ------------------------------------------------------------------------------------------
#
# PyTorch 모델은 학습한 매개변수를 ``state_dict``\ 라고 불리는 내부 상태 사전(internal state dictionary)에 저장합니다.
# 이 상태 값들은 ``torch.save`` 메소드를 사용하여 저장(persist)할 수 있습니다:

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

##########################
# 모델 가중치를 불러오기 위해서는, 먼저 동일한 모델의 인스턴스(instance)를 생성한 다음에 ``load_state_dict()`` 메소드를 사용하여
# 매개변수들을 불러옵니다.

model = models.vgg16() # 기본 가중치를 불러오지 않으므로 pretrained=True를 지정하지 않습니다.
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

###########################
# .. note:: 추론(inference)을 하기 전에 ``model.eval()`` 메소드를 호출하여 드롭아웃(dropout)과 배치 정규화(batch normalization)를 평가 모드(evaluation mode)로 설정해야 합니다. 그렇지 않으면 일관성 없는 추론 결과가 생성됩니다.

#######################################################################
# 모델의 형태를 포함하여 저장하고 불러오기
# ------------------------------------------------------------------------------------------
#
# 모델의 가중치를 불러올 때, 신경망의 구조를 정의하기 위해 모델 클래스를 먼저 생성(instantiate)해야 했습니다.
# 이 클래스의 구조를 모델과 함께 저장하고 싶으면, (``model.state_dict()``\ 가 아닌) ``model`` 을 저장 함수에
# 전달합니다:

torch.save(model, 'model.pth')

########################
# 다음과 같이 모델을 불러올 수 있습니다:

model = torch.load('model.pth')

########################
# .. note:: 이 접근 방식은 Python `pickle <https://docs.python.org/3/library/pickle.html>`_ 모듈을 사용하여 모델을 직렬화(serialize)하므로, 모델을 불러올 때 실제 클래스 정의(definition)를 적용(rely on)합니다.

#######################
# 관련 튜토리얼
# -----------------
# :doc:`/recipes/recipes/saving_and_loading_a_general_checkpoint`
