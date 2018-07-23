# -*- coding: utf-8 -*-
"""
PyTorch를 이용한 신경망-변환(Neural-Transfer)
======================================================
**저자**: `Alexis Jacq <https://alexis-jacq.github.io>`_
  **번역**: `김봉모 <http://fmttm.egloos.com>`_

소개
------------------

환영합니다!. 이 문서는 Leon A. Gatys와 Alexander S. Ecker, Matthias Bethge 가 개발한
알고리즘인 `Neural-Style <https://arxiv.org/abs/1508.06576>`__ 를 구현하는 방법에 대해
설명하는 튜토리얼입니다.

신경망 뭐라고?
~~~~~~~~~~~~~~~~~~~

신경망 스타일(Neural-Style), 혹은 신경망 변화(Neural-Transfer)는 콘텐츠 이미지(예, 거북이)와 
스타일 이미지(예, 파도를 그린 예술 작품) 을 입력으로 받아 콘텐츠 이미지의 모양대로 스타일 이미지의
'그리는 방식'을 이용해 그린 것처럼 결과를 내는 알고리즘입니다:

.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1

어떻게 동작합니까?
~~~~~~~~~~~~~~~~~~~~~~~

원리는 간단합니다. 2개의 거리(distance)를 정의합니다. 하나는 콘텐츠( :math:`D_C` )를 위한 것이고 
다른 하나는 스타일( :math:`D_S` )을 위한 것입니다.
:math:`D_C` 는 콘텐츠 이미지와 스타일 이미지 간의 콘텐츠가 얼마나 차이가 있는지 측정을 합니다. 
반면에, :math:`D_S` 는 콘텐츠 이미지와 스타일 이미지 간의 스타일에서 얼마나 차이가 있는지를 측정합니다.
그런 다음, 세 번째 이미지를 입력(예, 노이즈로 구성된 이미지)으로부터 콘텐츠 이미지와의 콘텐츠 거리 
및 스타일 이미지와의 스타일 거리를 최소화하는 방향으로 세 번째 이미지를 변환합니다.

그래서. 어떻게 동작하냐고요?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

자, 더 나아가려면 수학이 필요합니다. :math:`C_{nn}` 를 사전 훈련된 깊은 합성곱 신경망 
네트워크(pre-trained deep convolutional neural network)라고 하고, :math:`X` 를 어떤 이미지라고 해보겠습니다.
:math:`C_{nn}(X)` 은 입력 이미지 X를 입력으로 해서 CNN 을 통과한 네트워크(모든 레이어들의 특징 맵(feature map)을 포함하는)를 의미합니다.
:math:`F_{XL} \in C_{nn}(X)` 는 깊이 레벨 L에서의 특징 맵(feature map)을 의미하고, 
모두 벡터화(vectorized)되고 연결된(concatenated) 하나의 단일 벡터입니다.
그리고, :math:`Y` 를 이미지 :math:`X` 와 크기가 같은 이미지라고 하면, 
레이어 :math:`L` 에 해당하는 콘텐츠의 거리를 정의할 수 있습니다:

.. math:: D_C^L(X,Y) = \|F_{XL} - F_{YL}\|^2 = \sum_i (F_{XL}(i) - F_{YL}(i))^2

:math:`F_{XL}(i)` 는 :math:`F_{XL}` 의 :math:`i^{번째}` 요소(element) 입니다.
스타일에 해당하는 내용은 위 내용보다 조금 더 신경 쓸 부분이 있습니다.
:math:`F_{XL}^k` 를 레이어 :math:`L` 에서 특징 맵(feature map) :math:`K` 의 :math:`k^{번째}` 에 해당하는
벡터화된 :math:`k \leq K` 라고 해 보겠습니다.
스타일 :math:`G_{XL}` 의 :math:`X` 레이어에서 :math:`L` 은 모든 벡터화된 특징 맵(feature map) :math:`F_{XL}^k` 
에서 :math:`k \leq K` 그람(Gram)으로 정의 됩니다.
다시 말하면, :math:`G_{XL}` 는 :math:`K`\ x\ :math:`K` 행렬과 요소 :math:`G_{XL}(k,l)` 의 :math:`k^{번째}` 줄과
:math:`l^{번째}` 행의 :math:`G_{XL}` 는 :math:`F_{XL}^k` 와 :math:`F_{XL}^l` 간의
벡터화 곱을 의미합니다:

.. math::

    G_{XL}(k,l) = \langle F_{XL}^k, F_{XL}^l\\rangle = \sum_i F_{XL}^k(i) . F_{XL}^l(i)

:math:`F_{XL}^k(i)` 는 :math:`F_{XL}^k` 의 :math:`i^{번째}` 요소 입니다.
우리는 :math:`G_{XL}(k,l)` 를 특징 맵(feature map) :math:`k` 와 :math:`l` 간의 
상관 관계(correlation)에 대한 척도로 볼 수 있습니다.
그런 의미에서, :math:`G_{XL}` 는 특징 맵(feature map) :math:`X` 의 레이어 :math:`L` 에서의 
상관 관계 행렬을 나타냅니다.
:math:`G_{XL}` 의 크기는 단지 특징 맵(feature map)의 숫자에만 의존성이 있고,
:math:`X` 의 크기에는 의존성이 없다는 것을 유의 해야 합니다.
그러면, 만약 :math:`Y` 가 다른 *어떤 크기의* 이미지라면,
우리는 다음과 같이 레이어 :math:`L` 에서 스타일의 거리를 정의 합니다.

.. math::

    D_S^L(X,Y) = \|G_{XL} - G_{YL}\|^2 = \sum_{k,l} (G_{XL}(k,l) - G_{YL}(k,l))^2

:math:`D_C(X,C)` 의 한 번의 최소화를 위해서, 이미지 변수 :math:`X` 와 대상 콘텐츠-이미지 :math:`C` 와
:math:`D_S(X,S)` 와 :math:`X` 와 대상 스타일-이미지 :math:`S` , 둘 다 여러 레이어들에 대해서 계산되야 하고,
우리는 원하는 레이어 각각에서의 거리의 그라디언트를 계산하고 더합니다( :math:`X` 와 관련된 도함수):

.. math::

    \\nabla_{\textit{total}}(X,S,C) = \sum_{L_C} w_{CL_C}.\\nabla_{\textit{content}}^{L_C}(X,C) + \sum_{L_S} w_{SL_S}.\\nabla_{\textit{style}}^{L_S}(X,S)

:math:`L_C` 와 :math:`L_S` 는 각각 콘텐츠와 스타일의 원하는 (임의 상태의) 레이어들을 의미하고,
:math:`w_{CL_C}` 와 :math:`w_{SL_S}` 는 원하는 레이어에서의
스타일 또는 콘텐츠의 가중치를 (임의 상태의) 의미합니다.
그리고 나서, 우리는 :math:`X` 에 대해 경사 하강법을 실행합니다.

.. math:: X \leftarrow X - \\alpha \\nabla_{\textit{total}}(X,S,C)

네, 수학은 이정도면 충분합니다. 만약 더 깊이 알고 싶다면 (그레이언트를 어떻게 계산하는지),
Leon A. Gatys and AL가 작성한 **원래의 논문을 읽어 볼 것을 권장합니다** 
논문에는 앞서 설명한 내용들 모두에 대해 보다 자세하고 명확하게 얘기합니다.

구현을 위해서 PyTorch에서는 이미 우리가 필요로하는 모든 것을 갖추고 있습니다. 
실제로 PyTorch를 사용하면 라이브러리의 함수를 사용하는 동안 모든 그라디언트(Gradient)가 
자동,동적으로 계산됩니다.(라이브러리에서 함수를 사용하는 동안)
이런 점이 PyTorch에서 알고리즘 구현을 매우 편리하게 합니다.

PyTorch 구현
----------------------

위의 모든 수학을 이해할 수 없다면, 구현함으로써 이해도를 높여 갈 수 있을 것 입니다. 
PyTorch를 이용할 예정이라면, 먼저 이 문서 :doc:`Introduction to PyTorch </beginner/deep_learning_60min_blitz>` 를 읽어볼 것을 추천 합니다.

패키지들
~~~~~~~~

우리는 다음 패키지들을 활용 할 것입니다:

-  ``torch`` , ``torch.nn``, ``numpy`` (PyTorch로 신경망 처리를 위한 필수 패키지)
-  ``torch.optim`` (효율적인 그라디언트 디센트)
-  ``PIL`` , ``PIL.Image`` , ``matplotlib.pyplot`` (이미지를 읽고 보여주는 패키지)
-  ``torchvision.transforms`` (PIL타입의 이미지들을 토치 텐서 형태로 변형해주는 패키지)
-  ``torchvision.models`` (사전 훈련된 모델들의 학습 또는 읽기 패키지)
-  ``copy`` (모델들의 깊은 복사를 위한 시스템 패키지)
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


######################################################################
# 쿠다(CUDA)
# ~~~~~~~~~~~~~~
#
# 컴퓨터에 GPU가 있는 경우, 특히 VGG와 같이 깊은 네트워크를 사용하려는 경우 
# 알고리즘을 CUDA 환경에서 실행하는 것이 좋습니다. 
# CUDA를 쓰기 위해서 Pytorch에서는 ``torch.cuda.is_available()`` 를 제공하는데, 
# 작업하는 컴퓨터에서 GPU 사용이 가능하면 ``True`` 를 리턴 합니다.
# 이후로, 우리는 ``.cuda()`` 라는 메소드를 사용하여 모듈과 관련된 할당된 프로세스를 CPU에서 GPU로 수 있습니다.
# 이 모듈을 CPU로 되돌리고 싶을 때에는 (예 : numpy에서 사용), 우리는 ``.cpu ()`` 메소드를 사용하면 됩니다.
# 마지막으로, ``.type(dtype)`` 메소드는 ``torch.FloatTensor`` 타입을 
# GPU에서 사용 할 수 있도록 ``torch.cuda.FloatTensor`` 로 변환하는데 사용할 수 있습니다.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# 이미지 읽기
# ~~~~~~~~~~~~~
#
# 구현을 간단하게 하기 위해서, 스타일 이미지와 콘텐츠 이미지의 크기를 동일하게 맞추어서 시작합니다.
# 그런 다음 원하는 출력 이미지 크기로 확장 시킵니다.(본 예제에서는 128이나 512로 하는데 GPU가 가능한 상황에 맞게 선택해서 하세요.)
# 그리고 영상 데이터를 토치 텐서로 변환하고, 신경망 네트워크에 사용할 수 있도록 준비합니다.
#
# .. Note::
#     튜토리얼을 실행하는 데 필요한 이미지를 다운로드하는 링크는 다음과 같습니다.:
#     `picasso.jpg <http://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ 와
#     `dancing.jpg <http://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     위 두개의 이미지를 다운로드 받아 디렉토리 이름 ``images`` 에 추가하세요.


# 출력 이미지의 원하는 크기를 정하세요.
imsize = 512 if torch.cuda.is_available() else 128  # gpu가 없다면 작은 크기로

loader = transforms.Compose([
    transforms.Resize(imsize),  # 입력 영상 크기를 맞춤
    transforms.ToTensor()])  # 토치 텐서로 변환


def image_loader(image_name):
    image = Image.open(image_name)
    # 네트워크의 입력 차원을 맞추기 위해 필요한 가짜 배치 차원
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("images/picasso.jpg")
content_img = image_loader("images/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


######################################################################
# 가져온 PIL 이미지는 0에서 255 사이의 이미지 픽셀값을 가집니다. 
# 토치 텐서로 변환하면 0에서 1의 값으로 변환됩니다. 
# 이는 중요한 디테일로: 토치 라이브러리의 신경망은 0에서 1의 텐서 이미지로 학습하게 됩니다.
# 0-255 텐서 이미지를 네트워크에 공급 하려고 하면 활성화된(activated) 특징 맵(feature map)은 의미가 없습니다.(역자주, 입력 값에 따라 RELU와 같은 활성화 레이어에서 입력으로 되는 값의 범위가 완전히 다르기 때문)
# Caffe 라이브러리의 사전 훈련된 네트워크의 경우는 그렇지 않습니다: 해당 모델들은 0에서 255 사이 값의 텐서 이미지로 학습 되었습니다.
#
# 이미지 표시하기
# ~~~~~~~~~~~~~~~~~~~~
#
# 우리는 이미지를 표시하기 위해 ``plt.imshow`` 를 이용합니다. 
# 그러기 위해 우선 텐서를 PIL 이미지로 변환해 주겠습니다:
#

unloader = transforms.ToPILImage()  # PIL 이미지로 재변환 합니다

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # 텐서의 값에 변화가 적용되지 않도록 텐서를 복제합니다
    image = image.squeeze(0)      # 페이크 배치 차원을 제거 합니다
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # 그리는 부분이 업데이트 될 수 있게 잠시 정지합니다


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


######################################################################
# 콘텐츠 로스
# ~~~~~~~~~~~~
#
# 콘텐츠 로스는 네트워크에서 :math:`X` 로 입력을 받았을 때 레이어 :math:`L` 에서 특징 맵(feature map) :math:`F_{XL}` 을 입력으로 가져 와서 
# 이 이미지와 콘텐츠 이미지 사이의 가중치 콘텐츠 거리 :math:`w_{CL}.D_C^L(X,C)` 를 반환하는 기능입니다. 
# 따라서, 가중치 :math:`w_{CL}` 및 목표 콘텐츠 :math:`F_{CL}` 은 함수의 파라미터 입니다.
# 우리는 이 매개 변수를 입력으로 사용하는 생성자(constructor)가 있는 토치 모듈로 함수를 구현합니다. 
# 거리 :math:`\|F_{XL} - F_{YL}\|^2` 는 세 번째 매개 변수로 명시된 기준 ``nn.MSELoss`` 를 사용하여
# 계산할 수 있는 두 세트의 특징 맵(feature map) 사이의 평균 제곱 오차(MSE, Mean Square Error)입니다.
#
# 우리는 신경망의 추가 모듈로서 각 레이어에 컨텐츠 로스를 추가 할 것 입니다. 
# 이렇게 하면 입력 영상 :math:`X` 를 네트워크에 보낼 때마다 원하는 모든 레이어에서 
# 모든 컨텐츠 로스가 계산되고 자동 그라디언트로 인해 모든 그라디언트가 계산됩니다. 
# 이를 위해 우리는 입력을 리턴하는 ``forward`` 메소드를 만들기만 하면 됩니다: 모듈은 신경망의 ''투명 레이어'' 가 됩니다. 
# 계산된 로스는 모듈의 매개 변수로 저장됩니다.
#
# 마지막으로 그라디언트를 재구성하기 위해 nn.MSELoss의 ``backward`` 메서드를 호출하는 가짜 backward 메서드를 정의 합니다. 
# 이 메서드는 계산된 로스를 반환 합니다. 이는 스타일 및 콘텐츠 로스의 진화를 표시하기 위해 그라디언트 디센트를 실행할 때 유용합니다.
#

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # 그라디언트를 동적으로 계산하는 데 사용되는 트리에서 대상 콘텐츠를 '분리' 합니다.
        # :이 값은 변수(variable)가 아니라 명시된 값입니다. 
        # 그렇지 않으면 기준의 전달 메소드가 오류를 발생 시킵니다.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


######################################################################
# .. Note::
#    **중요한 디테일**: 이 모듈은 ``ContentLoss`` 라고 이름 지어졌지만 진정한 PyTorch Loss 함수는 아닙니다. 컨텐츠 손실을 PyTorch Loss로 정의 하려면 PyTorch autograd Function을 생성 하고 ``backward`` 메소드에서 직접 그라디언트를 재계산/구현 해야 합니다.
#
# 스타일 로스
# ~~~~~~~~~~~~~~~~~~
#
# 스타일 손실을 위해 우리는 레이어 :math:`L` 에서 :math:`X` 로 공급된(입력으로 하는) 신경망의 특징 맵(feature map) :math:`F_{XL}` 이 주어진 경우
# 그램 생성 :math:`G_{XL}` 을 계산하는 모듈을 먼저 정의 해야 합니다. 
# :math:`\hat{F}_{XL}` 을 KxN 행렬에 대한 :math:`F_{XL}`의 모양을 변경한 버전이라고 하겠습니다.
# 여기서, :math:`K`는 레이어 :math:`L`에서의 특징 맵(feature map)들의 수이고, :math:`N` 은 임의의 벡터화 된 특징 맵(feature map) :math:`F_{XL}^k` 의 길이가 됩니다. 
# :math:`F_{XL}^k` 의 :math:`k^{번째}` 번째 줄은 :math:`F_{XL}^k` 입니다. 
# math:`\hat{F}_{XL} \cdot \hat{F}_{XL}^T = G_{XL}` 인지 확인 해보길 바랍니다. 
# 이를 확인해보면 모듈을 구현하는 것이 쉬워 집니다:
#

def gram_matrix(input):
    a, b, c, d = input.size()  # a=배치 크기(=1)
    # b=특징 맵의 크기
    # (c,d)=특징 맵(N=c*d)의 차원

    features = input.view(a * b, c * d)  # F_XL을 \hat F_XL로 크기 조정합니다

    G = torch.mm(features, features.t())  # 그램 곱을 수행합니다

    # 그램 행렬의 값을 각 특징 맵의 요소 숫자로 나누는 방식으로 '정규화'를 수행합니다.
    return G.div(a * b * c * d)


######################################################################
# 특징 맵(feature map) 차원 :math:`N`이 클수록, 그램(Gram) 행렬의 값이 커집니다. 
# 따라서 :math:`N`으로 정규화하지 않으면 첫번째 레이어에서 계산된 로스 (풀링 레이어 전에)는
# 경사 하강법 동안 훨씬 더 중요하게 됩니다. (역자주 : 정규화를 하지 않으면 첫번째 레이어에서 계산된 값들의 가중치가 높아져 상대적으로 다른 레이어에서 계산한 값들의 반영이 적게 되버리기 때문에 정규화가 필요해집니다.)
# 스타일 특징의 흥미로운 부분들은 가장 깊은 레이어에 있기 때문에 그렇게 동작하지 않도록 해야 합니다!
#
# 그런 다음 스타일 로스 모듈은 콘텐츠 로스 모듈과 완전히 동일한 방식으로 구현되지만
# 대상과 입력 간의 그램 매트릭스의 차이를 비교하게 됩니다
#

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


######################################################################
# 뉴럴 네트워크 읽기
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# 자, 우리는 사전 훈련된 신경망을 가져와야 합니다. 이 논문에서와 같이, 
# 우리는 19 레이어 층을 가지는 VGG(VGG19) 네트워크를 사전 훈련된 네트워크로 사용할 것입니다.
#
# PyTorch의 VGG 구현은 두 개의 하위 순차 모듈로 나뉜 모듈 입니다. 
# ``특징(features)`` 모듈 : 합성곱과 풀링 레이어들을 포함 합니다.
# ``분류(classifier)`` 모듈 : fully connected 레이어들을 포함 합니다.
# 우리는 여기서 ``특징`` 모듈에 관심이 있습니다.
# 일부 레이어는 학습 및 평가에 있어서 상황에 따라 다른 동작을 합니다. 
# 이후 우리는 그것을 특징 추출자로 사용하고 있습니다. 
# 우리는 .eval() 을 사용하여 네트워크를 평가 모드로 설정 할 수 있습니다.
#

cnn = models.vgg19(pretrained=True).features.to(device).eval()

######################################################################
# 또한 VGG 네트워크는 평균 = [0.485, 0.456, 0.406] 및 표준편차 = [0.229, 0.224, 0.225]로 정규화 된 각 채널의 이미지에 대해 학습된 모델입니다.
# (역자, 일반적으로 네트워크는 이미지넷으로 학습이 되고 이미지넷 데이터의 평균과 표준편차가 위의 값과 같습니다.)
# 우리는 입력 이미지를 네트워크로 보내기 전에 정규화 하는데 위 평균과 표준편차 값을 사용합니다.
#

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 입력 이미지를 정규화하는 모듈을 만들어 nn.Sequential에 쉽게 입력 할 수 있게 하세요.
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view(텐서의 모양을 바꾸는 함수)로 평균과 표준 편차 텐서를 [C x 1 x 1] 형태로 만들어
        # 바로 입력 이미지 텐서의 모양인 [B x C x H x W] 에 연산할 수 있도록 만들어 주세요.
        # B는 배치 크기, C는 채널 값, H는 높이, W는 넓이 입니다.

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # img 값 정규화(normalize)
        return (img - self.mean) / self.std


######################################################################
# ``순차(Sequential)`` 모듈에는 하위 모듈의 정렬된 목록이 있습니다. 
# 예를 들어 ``vgg19.features`` 은 vgg19 구조의 올바른 순서로 정렬된 순서 정보(Conv2d, ReLU, MaxPool2d, Conv2d, ReLU ...)를 포함합니다. 
# 콘텐츠 로스 섹션에서 말했듯이 우리는 네트워크의 원하는 레이어에 추가 레이어 '투명(transparent)'레이어로 스타일 및 콘텐츠 손실 모듈을 추가하려고 합니다. 
# 이를 위해 새로운 순차 모듈을 구성합니다.이 모듈에서는 vgg19의 모듈과 손실 모듈을 올바른 순서로 추가합니다.
#

# 스타일/콘텐츠 로스로 계산하길 원하는 깊이의 레이어들:
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # 표준화(normalization) 모듈
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # 단지 반복 가능한 접근을 갖거나 콘텐츠/스타일의 리스트를 갖기 위함
    # 로스값
    content_losses = []
    style_losses = []

    # cnn은 nn.Sequential 하다고 가정하므로, 새로운 nn.Sequential을 만들어
    # 우리가 순차적으로 활성화 하고자하는 모듈들을 넣겠습니다.
    model = nn.Sequential(normalization)

    i = 0  # conv레이어를 찾을때마다 값을 증가 시킵니다
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # in-place(입력 값을 직접 업데이트) 버전은 콘텐츠로스와 스타일로스에
            # 좋은 결과를 보여주지 못합니다.
            # 그래서 여기선 out-of-place로 대체 하겠습니다.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # 콘텐츠 로스 추가:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # 스타일 로스 추가:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 이제 우리는 마지막 콘텐츠 및 스타일 로스 이후의 레이어들을 잘라냅니다.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


######################################################################
# .. Note::
#    논문에서는 맥스 풀링(Max Pooling) 레이어를 에버리지 풀링(Average Pooling) 레이어로 바꾸는 것을 추천합니다.
#    AlexNet에서는 논문에서 사용된 VGG19 네트워크보다 상대적으로 작은 네트워크라 결과 품질에서 
#    큰 차이를 확인하기 어려울 수 있습니다.
#    그러나, 만약 당신이 대체해 보기를 원한다면 아래 코드들을 사용할 수 있습니다:
#
#    ::
#
#        # avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size,
#        #                         stride=layer.stride, padding = layer.padding)
#        # model.add_module(name,avgpool)


######################################################################
# 입력 이미지
# ~~~~~~~~~~~~~~~~~~~
#
# 다시, 코드를 간단하게 하기 위해, 콘텐츠와 스타일 이미지들의 같은 차원의 이미지를 가져옵니다.
# 해당 이미지는 백색 노이즈일 수 있거나 콘텐츠-이미지의 값들을 복사해도 좋습니다.
#

input_img = content_img.clone()
# 대신에 백색 노이즈를 이용하길 원한다면 아래 줄의 주석처리를 제거하세요:
# input_img = torch.randn(content_img.data.size(), device=device)

# 원본 입력 이미지를 창에 추가합니다:
plt.figure()
imshow(input_img, title='Input Image')


######################################################################
# 경사 하강법
# ~~~~~~~~~~~~~~~~
#
# 알고리즘의 저자인 Len Gatys 가 `여기서 <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__ 제안한 방식대로
# 경사 하강법을 실행하는데 L-BFGS 알고리즘을 사용 하겠습니다.
# 일반적인 네트워크 학습과는 다르게, 우리는 콘텐츠/스타일 로스를 최소화 하는 방향으로 입력 영상을 학습 시키려고 합니다.
# 우리는 간단히 PyTorch L-BFGS 옵티마이저 ``optim.LBFGS`` 를 생성하려고 하며, 최적화를 위해 입력 이미지를 텐서 타입으로 전달합니다. 
# 우리는 ``.requires_grad_()`` 를 사용하여 해당 이미지가 그라디언트가 필요함을 확실하게 합니다.
#

def get_input_optimizer(input_img):
    # 이 줄은 입력은 그레이던트가 필요한 파라미터라는 것을 보여주기 위해 있습니다.
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


######################################################################
# **마지막 단계**: 경사 하강의 반복. 각 단계에서 우리는 네트워크의 새로운 로스를 계산하기 위해
# 업데이트 된 입력을 네트워크에 공급해야 합니다. 우리는 그라디언트를 동적으로 계산하고 
# 그라디언트 디센트의 단계를 수행하기 위해 각 손실의 ``역방향(backward)`` 메소드를 실행해야 합니다.
# 옵티마이저는 인수로서 "클로저(closure)"를 필요로 합니다: 즉, 모델을 재평가하고 로스를 반환 하는 함수입니다.
#
# 그러나, 여기에 작은 함정이 있습니다. 최적화 된 이미지는 0 과 1 사이에 머물지 않고 :math:`-\infty`과 :math:`+\infty` 사이의 값을 가질 수 있습니다. 
# 다르게 말하면, 이미지는 잘 최적화될 수 있고(0-1 사이의 정해진 값 범위내의 값을 가질 수 있고) 이상한 값을 가질 수도 있습니다. 
# 사실 우리는 입력 이미지가 올바른 범위의 값을 유지할 수 있도록 제약 조건 하에서 최적화를 수행해야 합니다. 
# 각 단계마다 0-1 간격으로 값을 유지하기 위해 이미지를 수정하는 간단한 해결책이 있습니다.
#

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """스타일 변환을 실행합니다."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # 입력 이미지의 업데이트된 값들을 보정합니다
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # 마지막 보정...
    input_img.data.clamp_(0, 1)

    return input_img

######################################################################
# 마지막으로, 알고리즘을 실행 시킵니다.

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
