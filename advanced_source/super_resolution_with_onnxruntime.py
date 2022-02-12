"""
(선택) PyTorch 모델을 ONNX으로 변환하고 ONNX 런타임에서 실행하기
========================================================================
이 튜토리얼에서는 어떻게 PyTorch에서 정의된 모델을 ONNX 형식으로 변환하고 또 어떻게 그 변환된 모델을
ONNX 런타임에서 실행할 수 있는지에 대해 알아보도록 하겠습니다.
ONNX 런타임은 ONNX 모델을 위한 엔진으로서 성능에 초점을 맞추고 있고 여러 다양한 플랫폼과 하드웨어(윈도우,
리눅스, 맥을 비롯한 플랫폼 뿐만 아니라 CPU, GPU 등의 하드웨어)에서 효율적인 추론을 가능하게 합니다.
ONNX 런타임은 `여기
<https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-runtime-machine-learning-inferencing-0-4-release>`__ 에서
설명된 것과 같이 여러 모델들의 성능을 상당히 높일 수 있다는 점이 증명되었습니다.
이 튜토리얼을 진행하기 위해서는 `ONNX <https://github.com/onnx/onnx>`__
와 `ONNX Runtime <https://github.com/microsoft/onnxruntime>`__ 설치가 필요합니다.
ONNX와 ONNX 런타임의 바이너리 빌드를 ``pip install onnx onnxruntime`` 를 통해 받을 수 있습니다.
ONNX 런타임은 버전 3.5에서 3.7까지의 Python과 호환됩니다.
``참고``: 본 튜토리얼은 PyTorch의 master 브랜치를 필요로하며 `링크 <https://github.com/pytorch/pytorch#from-source>`__ 에서
설치할 수 있습니다.
"""

# 필요한 import문
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


######################################################################
# 초해상화(super-resolution)란 이미지나 비디오의 해상도를 높이기 위한 방법으로 이미지 프로세싱이나 비디오 편집에 널리
# 사용되고 있는 방법입니다. 이 튜토리얼에서는 크기가 작은 초해상화 모델을 사용하도록 하겠습니다.
#
# 먼저, 초해상화 모델을 PyTorch에서 구현하겠습니다.
# 이 모델은 `"Real-Time Single Image and Video Super-Resolution Using an Efficient
# Sub-Pixel Convolutional Neural Network" - Shi et al <https://arxiv.org/abs/1609.05158>`__ 에서 소개된
# 효율적인 서브픽셀 합성곱 계층을 사용하여 이미지의 해상도를 업스케일 인자만큼 늘립니다.
# 모델은 이미지의 YCbCr 성분 중 Y 성분을 입력값으로 받고 업스케일된 초해상도의 Y 채널 값을 리턴합니다.
#
# 아래는 PyTorch 예제의
# `모델
# <https://github.com/pytorch/examples/blob/master/super_resolution/model.py>`__
# 을 그대로 가져온 것입니다:
#

# PyTorch에서 구현된 초해상도 모델
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# 위에서 정의된 모델을 사용하여 초해상도 모델 생성
torch_model = SuperResolutionNet(upscale_factor=3)


######################################################################
# 위 과정이 끝나면 일반적인 경우에 모델을 학습시키기 시작할 것입니다. 하지만 본 튜토리얼에서는
# 미리 학습된 가중치들을 사용하도록 하겠습니다. 참고로 이 모델은 높은 정확도에 이를 때까지 학습되지 않았고
# 본 튜토리얼을 원활히 진행하기 위한 목적으로 사용하는 것입니다.
#
# 모델을 변환하기 전에 모델을 추론 모드로 바꾸기 위해서 ``torch_model.eval()`` 또는 ``torch_model.train(False)`` 를
# 호출하는 것이 중요합니다.
# 이는 dropout이나 batchnorm과 같은 연산들이 추론과 학습 모드에서 다르게 작동하기 때문에 필요합니다.
#

# 미리 학습된 가중치를 읽어옵니다
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # 임의의 수

# 모델을 미리 학습된 가중치로 초기화합니다
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# 모델을 추론 모드로 전환합니다
torch_model.eval()


######################################################################
# 이제 Tracing이나 스크립팅을 통해서 PyTorch 모델을 변환할 수 있습니다.
# 이 튜토리얼에서는 tracing을 통해 변환된 모델을 사용하도록 하겠습니다.
# 모델을 변환하기 위해서는 ``torch.onnx.export()`` 함수를 호출합니다.
# 이 함수는 모델을 실행하여 어떤 연산자들이 출력값을 계산하는데 사용되었는지를 기록합니다.
# ``export`` 함수가 모델을 실행하기 때문에, 우리가 직접 텐서를 입력값으로 넘겨주어야 합니다.
# 이 텐서의 값은 알맞은 자료형과 모양이라면 랜덤하게 결정되어도 무방합니다.
# 특정 차원을 동적 차원으로 지정하지 않는 이상, ONNX로 변환된 그래프의 경우 입력값의 크기는 모든 차원에 대해 고정됩니다.
# 예시에서는 모델이 항상 배치 사이즈 1을 사용하도록 변환하였지만, ``torch.onnx.export()`` 의 ``dynamic_axes`` 인자의
# 첫번째 차원은 동적 차원으로 지정합니다. 따라서 변환된 모델은 임의의 batch_size에 대해 [batch_size, 1, 224, 224] 사이즈
# 입력값을 받을 수 있습니다. 
#
# PyTorch의 변환 인터페이스에 대해 더 자세히 알고 싶다면
# `torch.onnx 문서 <https://pytorch.org/docs/master/onnx.html>`__ 를 참고해주세요.
#

# 모델에 대한 입력값
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# 모델 변환
torch.onnx.export(torch_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "super_resolution.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

######################################################################
# ONNX 런타임에서 변환된 모델을 사용했을 때 같은 결과를 얻는지 확인하기 위해서 ``torch_out`` 를 계산합니다.
#
# ONNX 런타임에서의 모델 결과값을 확인하기 전에 먼저 ONNX API를 사용해 ONNX 모델을 확인해보도록 하겠습니다.
# 먼저, ``onnx.load("super_resolution.onnx")`` 는 저장된 모델을 읽어온 후
# 머신러닝 모델을 취합하여 저장하고 있는 상위 파일 컨테이너인 onnx.ModelProto를 리턴합니다.
# onnx.ModelProto에 대해 더 자세한 것은 `onnx.proto 기술문서 <https://github.com/onnx/onnx/blob/master/onnx/onnx.proto>`__ 에서
# 확인하실 수 있습니다.
# ``onnx.checker.check_model(onnx_model)`` 는 모델의 구조를 확인하고
# 모델이 유효한 스키마(valid schema)를 가지고 있는지를 체크합니다.
# ONNX 그래프의 유효성은 모델의 버전, 그래프 구조, 노드들, 그리고 입력값과 출력값들을 모두 체크하여
# 결정됩니다.
#

import onnx

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)


######################################################################
# 이제 ONNX 런타임의 Python API를 통해 결과값을 계산해보도록 하겠습니다.
# 이 부분은 보통 별도의 프로세스 또는 별도의 머신에서 실행되지만, 이 튜토리얼에서는 모델이
# ONNX 런타임과 PyTorch에서 동일한 결과를 출력하는지를 확인하기 위해 동일한 프로세스에서 계속 실행하도록
# 하겠습니다.
#
# 모델을 ONNX 런타임에서 실행하기 위해서는 미리 설정된 인자들(본 예제에서는 기본값을 사용합니다)로
# 모델을 위한 추론 세션을 생성해야 합니다.
# 세션이 생성되면, 모델의 run() API를 사용하여 모델을 실행합니다.
# 이 API의 리턴값은 ONNX 런타임에서 연산된 모델의 결과값들을 포함하고 있는 리스트입니다.
#

import onnxruntime

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")


######################################################################
# 이제 PyTorch와 ONNX 런타임에서 연산된 결과값이 서로 일치하는지 오차범위 (rtol=1e-03, atol=1e-05)
# 이내에서 확인해야 합니다.
# 만약 결과가 일치하지 않는다면 ONNX 변환기에 문제가 있는 것이니 저희에게 알려주시기 바랍니다.
#


######################################################################
# ONNX 런타임에서 이미지를 입력값으로 모델을 실행하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# 지금까지 PyTorch 모델을 변환하고 어떻게 ONNX 런타임에서 구동하는지 가상의 텐서를 입력값으로 하여
# 살펴보았습니다.

######################################################################
# 본 튜토리얼에서는 아래와 같은 유명한 고양이 사진을 사용하도록 하겠습니다.
#
# .. figure:: /_static/img/cat_224x224.jpg
#    :alt: cat
#

######################################################################
# 먼저, PIL 라이브러리를 사용하여 이미지를 로드하고 전처리하겠습니다. 이 전처리는 신경망 학습과 테스트에
# 보편적으로 적용되고 있는 전처리 과정입니다.
#
# 먼저 이미지를 모델의 입력값 크기(224x224)에 맞게 리사이즈합니다.
#  그리고 이미지를 Y, Cb, Cr 성분으로 분해합니다.
# Y 성분[역자 주: 휘도 성분]은 그레이스케일(회색조) 이미지를 나타내고, Cb 성분은 파란색에서 밝기를 뺀 색차 성분,
# Cr은 빨강색에서 밝기를 뺀 색차 성분을 나타냅니다.
# 사람의 눈은 Y 성분에 더 민감하게 반응하기 때문에 저희에게는 현재 이 성분이 중요하고, 이 Y 성분을 변환할 것입니다.
# Y 성분을 뽑아낸 뒤에, 추출한 Y 성분을 모델의 입력값이 될 텐서로 변환합니다.
#

from PIL import Image
import torchvision.transforms as transforms

img = Image.open("./_static/img/cat.jpg")

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)


######################################################################
# 다음은 리사이즈된 그레이스케일 고양이 이미지 텐서를 앞서 설명했던 것처럼 초해상도 모델에 넘겨주어
# ONNX 런타임에서 실행하도록 하겠습니다.
#

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]


######################################################################
# 이렇게 하면 모델의 결과값은 텐서가 됩니다.
# 이제 모델의 결과값을 처리하여 결과값 텐서에서 마지막 최종 출력 이미지를 만들고 이를 저장해보도록 하겠습니다.
# 후처리 단계는 `링크 <https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py>`__ 에
# 구현되어있는 초해상도 모델 코드에서 가져온 것입니다.
#

img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# PyTorch 버전의 후처리 과정 코드를 이용해 결과 이미지 만들기
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# 이미지를 저장하고 모바일 기기에서의 결과 이미지와 비교하기
final_img.save("./_static/img/cat_superres_with_ort.jpg")


######################################################################
# .. figure:: /_static/img/cat_superres_with_ort.jpg
#    :alt: output\_cat
#
#
# ONNX 런타임은 크로스플랫폼 엔진으로서 CPU와 GPU 뿐만 아니라 여러 플랫폼에서 실행될 수 있습니다.
#
# ONNX 런타임은 Azure Machine Learning Services와 같은 클라우드에 배포되어 모델 추론을 하는데
# 사용될 수도 있습니다.
# 더 자세한 내용은 `링크 <https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx>`__ 를
# 참조해주십시오.
#
# ONNX 런타임의 성능에 관한 것은 `여기 <https://github.com/microsoft/onnxruntime#high-performance>`__ 에서
# 확인하실 수 있습니다.
#
# ONNX 런타임에 관한 더 자세한 내용은 `이 곳 <https://github.com/microsoft/onnxruntime>`__ 을 참조해 주세요.
#
