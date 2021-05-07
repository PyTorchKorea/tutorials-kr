"""
PyTorch에서 다양한 장치 간 모델을 저장하고 불러오기
===================================================

다양한 장치(device)에서 당신의 신경망 모델을 저장하거나 불러오고 싶은 
경우가 생길 수 있습니다.

개요
------------

PyTorch를 사용하여 장치 간의 모델을 저장하거나 불러오는 것은 비교적 간단합니다.
이번 레시피에서는, CPU와 GPU에서 모델을 저장하고 불러오는 방법을 실험할 것입니다.

설정
-----

이번 레시피에서 모든 코드 블록이 제대로 실행되게 하려면, 
우선 런타임(runtime) 설정을 "GPU"나 더 높게 지정해주어야 합니다. 
이후, 아래와 같이 ``torch``를 설치해야 PyTorch를 사용할 수 있습니다.

::

   pip install torch

"""


######################################################################
# 단계
# -----
# 
# 1. 데이터 활용에 필요한 모든 라이브러리 Import 하기
# 2. 신경망을 구성하고 초기화하기
# 3. GPU에서 저장하고 CPU에서 불러오기
# 4. GPU에서 저장하고 GPU에서 불러오기
# 5. CPU에서 저장하고 GPU에서 불러오기
# 6. ``DataParallel`` 모델을 저장하고 불러오기
# 
# 1. 데이터 활용에 필요한 모든 라이브러리 Import 하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 이번 레시피에서 우리는 ``torch`` 및 하위 패키지인 ``torch.nn``와 
# ``torch.optim``을 사용할 것입니다.
# 

import torch
import torch.nn as nn
import torch.optim as optim


######################################################################
# 2. 신경망을 구성하고 초기화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 예로, 이미지 트레이닝을 위한 신경망을 생성해보겠습니다.
# 자세한 내용은 신경망 정의 레시피를 참조하세요.
# 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

net = Net()
print(net)


######################################################################
# 3. GPU에서 저장하고 CPU에서 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# GPU로 학습된 모델을 CPU에서 불러올 때는 ``torch.load()`` 함수의 
# ``map_location`` 인자에 ``torch.device('cpu')``를 전달합니다.
# 

# 저장하고자 하는 경로를 지정합니다.
PATH = "model.pt"

# 저장하기
torch.save(net.state_dict(), PATH)

# 불러오기
device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load(PATH, map_location=device))


######################################################################
# 이 경우, Tensor의 저장된 내용은 ``map_location`` 인자를 통하여 CPU 장치에
# 동적으로 재배치됩니다.
# 
# 4. GPU에서 저장하고 GPU에서 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# GPU에서 학습하고 저장된 모델을 GPU에서 불러올 때는, 초기화된 모델에
# ``model.to(torch.device('cuda'))``을 호출하여 CUDA에 최적화된 모델로 
# 변환해주세요.
# 
# 그리고 모든 입력에 ``.to(torch.device('cuda'))`` 함수를 호출해야 
# 모델에 데이터를 제공할 수 있습니다.
# 

# 저장하기
torch.save(net.state_dict(), PATH)

# 불러오기
device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)


######################################################################
# ``my_tensor.to(device)``를 호출하면 GPU에 ``my_tensor``의 복사본이
# 반환되며, 이는 ``my_tensor``를 덮어쓰는 것이 아닙니다.
# 그러므로, Tensor를 직접 덮어써 주어야 한다는 것을 기억하세요:
# ``my_tensor = my_tensor.to(torch.device('cuda'))``.
# 
# 5. CPU에서 저장하고 GPU에서 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# CPU에서 학습하고 저장된 모델을 GPU에서 불러올 때는,``torch.load()``함수의 
# ``map_location``인자를 ``cuda:device_id``로 설정해주세요.
# 그러면 주어진 GPU 장치에 모델이 불러와 집니다.
# 
# 모델의 매개변수 Tensor를 CUDA Tensor로 변환하기 위해,
# ``model.to(torch.device('cuda'))``를 호출해주세요.
# 

# 저장하기
torch.save(net.state_dict(), PATH)

# 불러오기
device = torch.device("cuda")
model = Net()
# 사용하고자 하는 GPU 장치 번호를 지정합니다.
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
# 모델에 사용되는 모든 입력 Tensor들에 대해 input = input.to(device) 을 호출해야 합니다.
model.to(device)


######################################################################
# 6. ``torch.nn.DataParallel`` 모델을 저장하고 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# ``torch.nn.DataParallel``은 병렬 GPU 활용을 가능하게 하는 모델 래퍼(wrapper)입니다.
# 
# ``DataParallel`` 모델을 범용적으로 저장하기 위해서는
# ``model.module.state_dict()``을 사용하면 됩니다.
# 그러면 원하는 장치에 원하는 방식으로 유연하게 모델을 불러올 수 있습니다.
# 

# 저장하기
torch.save(net.module.state_dict(), PATH)

# 사용할 장치에 불러오기


######################################################################
# 축하합니다! PyTorch에서 다양한 장치 간에 모델을 성공적으로 저장하고 불러왔습니다.
# 
# 더 알아보기
# -----------
# 
# 다른 레시피를 둘러보고 계속 배워보세요:
# 
# -  TBD
# -  TBD
# 
