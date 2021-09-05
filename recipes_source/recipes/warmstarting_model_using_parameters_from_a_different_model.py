"""
PyTorch에서 다른 모델의 매개변수를 사용하여 빠르게 모델 시작하기(warmstart)
===========================================================================
모델을 부분적으로 불러오거나, 혹은 부분적인 모델을 불러오는 것은
학습 전이(Transfer learning)나 복잡한 모델을 새로 학습할 때 자주 접하는
시나리오입니다. 학습된 매개변수를 활용하면 학습 과정을 빠르게
시작(warmstart)할 수 있으며, 그러면 모델을 처음부터 훈련시킬 때보다 훨씬
일찍 수렴하리라 기대할 수 있습니다. 이는 활용할 수 있는 매개변수가 얼마 안
될 때에도 마찬가지입니다.

도입
----
일부 키가 누락된 부분적인 ``state_dict`` 를 불러올 때든, 아니면 결과를
저장할 모델보다 키가 많은 ``state_dict`` 를 불러올 때든,
``load_state_dict()`` 함수의 인자인 strict 를 False 로 두면 매치되지
않는 키를 무시하게끔 할 수 있습니다. 이 레시피에서는 다른 모델의
매개변수를 사용하여 모델을 빠르게 시작하는 실험을 진행해 보려 합니다.

설정
----
시작에 앞서서, ``torch`` 가 준비되어 있지 않다면 설치해야 합니다.

::

   pip install torch
   
"""



######################################################################
# 단계
# ----
# 
# 1. 데이터를 불러오는데 필요한 모든 라이브러리를 import 합니다
# 2. 신경망 A와 B를 정의하고 초기화합니다
# 3. 모델 A를 저장합니다
# 4. 모델 B로 모델을 불러옵니다
# 
# 1. 데이터를 불러올 때 필요한 라이브러리 import 하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 이 레시피에서는 ``torch`` 와, 그 하위 패키지인 ``torch.nn`` 및
# ``torch.optim`` 을 사용하겠습니다.
# 

import torch
import torch.nn as nn
import torch.optim as optim


######################################################################
# 2. 신경망 A와 B 정의하고 초기화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 하나의 예로써 이미지를 학습하는 신경망을 만들어 보려 합니다. 이에 대해
# 좀 더 알아보고 싶다면 신경망 정의하기에 대한 레시피를 참고하시기
# 바랍니다. 여기서는 신경망을 두 개 만들려고 하며, 신경망 A의매개변수를
# 신경망 B로 불러오려 합니다.
# 

class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

netA = NetA()

class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

netB = NetB()


######################################################################
# 3. 모델 A 저장하기
# ~~~~~~~~~~~~~~~~~~
# 

# 모델을 저장할 경로를 지정해 줍니다
PATH = "model.pt"

torch.save(netA.state_dict(), PATH)


######################################################################
# 4. 모델 B로 불러오기
# ~~~~~~~~~~~~~~~~~~~~
# 
# 한 레이어의 매개변수를 다른 레이어로 불러오려 하는데 일부 키가 매치되지
# 않는 상황이라고 해 봅시다. 그럴 때는 불러오려 하는 state_dict 의
# 매개변수 키의 이름을 바꿔서, 불러온 모델을 저장하려는 모델의 키와
# 매치되도록 해 주면 됩니다.
# 

netB.load_state_dict(torch.load(PATH), strict=False)


######################################################################
# 모든 키가 성공적으로 매치되었음을 확인할 수 있을 것입니다!
# 
# 축하합니다! 여러분은 PyTorch에서 다른 모델의 매개변수를 사용하여
# 모델을 빠르게 시작하는 방법에 대해 살펴보았습니다.
# 
# 좀 더 알아보기
# --------------
# 
# 계속 공부해 나가면서 다음 두 레시피를 살펴보기를 권합니다.
# 
# - `PyTorch에서 여러 모델을 하나의 파일에 저장하기 & 불러오기 <https://tutorials.pytorch.kr/recipes/recipes/saving_multiple_models_in_one_file.html>`__
# - `PyTorch에서 다양한 장치 간 모델을 저장하고 불러오기 <https://tutorials.pytorch.kr/recipes/recipes/save_load_across_devices.html>`__
