"""
Pytorch를 사용해 신경망 정의하기
====================================
딥러닝은 인공신경망(models)을 사용하며 이것은 상호연결된 집단의 많은 계층으로 구성된 계산 시스템입니다.
데이터가 이 상호연결된 집단을 통과하면서, 신경망은 입력을 출력으로 바꾸기 위해 요구된 계산 방법에 어떻게 근접하는 지를 배울 수 있습니다.
PyTorch에서, 신경망은 ``torch.nn`` 패키지를 사용해 구성할 수 있습니다.

소개
-----
PyTorch는 ``torch.nn`` 을 포함하여 신경망을 만들고 훈련시키는 것을 도울 수 있도록 섬세하게 만들어진 모듈과 클래스들을 제공합니다.
``nn.Moduel`` 은 계층, 그리고 ``output`` 을 반환하는 ``forward(input)`` 메소드를 포함하고 있습니다.

이 레시피에서, `MNIST dataset <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`__ 을 사용하여 신경망을 정의하기 위해 ``torch.nn`` 을 사용할 예정입니다.

설치
-----
시작하기 전에, 준비가 되어있지 않다면 ``torch`` 를 설치해야 합니다.

::

   pip install torch


"""


######################################################################
# 단계
# -----
#
# 1. 데이터를 가져오기 위해 필요한 라이브러리들 불러오기
# 2. 신경망을 정의하고 초기화하기
# 3. 데이터가 모델을 어떻게 지나갈 지 구체화하기
# 4. [선택사항] 데이터를 모델에 적용해 테스트하기
#
# 1. 데이터를 가져오기 위해 필요한 라이브러리들 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 레시피에서, ``torch`` 과 이것의 하위 모듈인 ``torch.nn`` , ``torch.nn.functional`` 을 사용합니다.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# 2. 신경망을 정의하고 초기화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이미지를 인식하는 신경망을 만들겁니다. PyTorch에서 만들어진 합성곱(convolution)이라고 불리는 방법을 사용하겠습니다.
# 합성곱은 커널이나 작은 행렬(matrix)를 통해 가중치를 부여한 이미지의 각 요소를 주변 값과 더합니다.
# 그리고 이것은 입력된 이미지의 특징(모서리 감지, 선명함, 흐릿함 등과 같은)을 추출하는 데 도움을 줍니다.
#
# 모델의 ``Net`` 클래스를 정의하기 위해 2가지가 필요합니다.
# 첫번째는 ``nn.Module`` 을 참고하는 ``__init__`` 함수를 작성하는 것입니다.
# 이 함수는 신경망에서 fully connected layers를 만드는 것에 사용됩니다.
#
# 합성곱을 사용해, 1개의 입력 이미지 채널을 가지고
# 목표인 0부터 9까지 숫자를 대표하는 10개의 라벨과 되응되 값을 출력하는 모델을 정의하겠습니다.
# 이 알고리즘은 만드는 사람에 달렸지만, 기본적인 MNIST 알고리즘을 따르도록 하겠습니다.
#

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # 첫번째 2D 합성곱 계층
      # 1개의 입력 채널(이미지)을 받아들이고, 사각 커널 사이즈가 3인 32개의 합성곱 특징들을 출력합니다.
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      # 두번째 2D 합성곱 계층
      # 32개의 입력 계층을 받아들이고, 사각 커널 사이즈가 3인 64개의 합성곱 특징을 출력합니다.
      self.conv2 = nn.Conv2d(32, 64, 3, 1)

      # 인접한 픽셀들은 입력 확률에 따라 모두 0 값을 가지거나 혹은 모두 유효한 값이 되도록 만듭니다.
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # 첫번째 fully connected layer
      self.fc1 = nn.Linear(9216, 128)
      # 10개의 라벨을 출력하는 두번째 fully connected layer
      self.fc2 = nn.Linear(128, 10)

my_nn = Net()
print(my_nn)


######################################################################
# 신경망을 정의하는 것을 마쳤습니다. 이제 어떻게 이것을 지나갈 지 정의해야 합니다.
#
# 3. 데이터가 모델을 어떻게 지나갈 지 구체화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch를 사용해 모델을 생성할 때, 계산 그래프(즉, 신경망)에 데이터를 지나가게 하는 ``forward`` 함수를 정의해야 합니다.
# 이것은 feed-forward 알고리즘을 나타냅니다.
#
# ``forward`` 함수에서 어떠한 Tensor 연산자도 사용 가능합니다.
#

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x는 데이터를 나타냅니다.
    def forward(self, x):
      # 데이터가 conv1을 지나갑니다.
      x = self.conv1(x)
      # x를 ReLU 활성함수(rectified-linear activation function)에 대입합니다.
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # x에 대해서 max pooling을 실행합니다.
      x = F.max_pool2d(x, 2)
      # 데이터가 dropout1을 지나갑니다.
      x = self.dropout1(x)
      # start_dim=1으로 x를 압축합니다.
      x = torch.flatten(x, 1)
      # 데이터가 fc1을 지나갑니다.
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # x에 softmax를 적용합니다.
      output = F.log_softmax(x, dim=1)
      return output


######################################################################
# 4. [선택사항] 데이터를 모델에 적용해 테스트하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 원하는 출력값을 받을 수 있는 지 확인하기 위해, 무작위의 데이터를 모델에 통과시켜 시험해봅시다.
#

# 임의의 28x28 이미지로 맞춰줍니다.
random_data = torch.rand((1, 1, 28, 28))

my_nn = Net()
result = my_nn(random_data)
print (result)


######################################################################
# 결과 tensor의 각 숫자는 임의의 tenosr와 연관된 라벨이 예측한 값과 같다는 것을 나타냅니다.
#
# 축하합니다! PyTorch로 신경망 정의하기를 성공적으로 해냈습니다.
#
# 더 알아보기
# -----------
#
# 계속해서 학습하고 싶다면 다른 레시피를 살펴보십시오:
#
# - `PyTorch에서 state_dict이 무엇인지 <https://tutorials.pytorch.kr/recipes/recipes/what_is_state_dict.html>`__
# - `PyTorch로 추론을 위한 모델을 저장하고 가저오기  <https://tutorials.pytorch.kr/recipes/recipes/saving_and_loading_models_for_inference.html>`__
