"""
PyTorch에서 변화도를 0으로 만들기
================================
신경망을 구축할 때는 변화도를 0으로 만들어 주는 것이 좋습니다. 기본적으로
``.backward()`` 를 호출할 때마다 변화도가 버퍼에 쌓이기 때문입니다. (덮어쓰지 않는다는 의미입니다.)

개요
------------
신경망을 학습시킬 때, 경사 하강법을 거쳐 모델 정확도를 높일 수 있습니다. 경사 하강법은 간단히
설명해 모델의 가중치와 편향을 약간씩 수정하면서 손실(또는 오류)를 최소화하는 과정입니다.

``torch.Tensor`` 는 PyTorch 의 핵심이 되는 클래스 입니다. 텐서를 생성할 때
``.requires_grad`` 속성을 ``True`` 로 설정하면, 텐서에 가해진 모든 연산을 추적합니다.
뒤따르는 모든 역전파 단계에서도 마찬가지로, 이 텐서의 변화도는 ``.grad`` 속성에 누적될 것입니다.
모든 변화도의 축적 또는 합은 손실 텐서에서 ``.backward()`` 를 호출할 때 계산됩니다.

텐서의 변화도를 0으로 만들어 주어야 하는 경우도 있습니다. 예를 들어 학습 과정 반복문을
시작할 때, 누적되는 변화도를 정확하게 추적하기 위해서는 변화도를 우선 0으로 만들어 주어야 합니다.
이 레시피에서는 PyTorch 라이브러리를 사용하여 변화도를 0으로 만드는 방법을 배워봅니다.
PyTorch에 내장된 ``CIFAR10`` 데이터셋에 대하여 신경망을 훈련시키는 과정을 통해 알아봅시다.

설정
-----
이 레시피에는 데이터를 학습시키는 내용이 포함되어 있기 때문에, 실행 가능한 노트북 파일이 있다면
런타임을 GPU 또는 TPU로 전환하는 것이 좋습니다. 시작하기에 앞서, ``torch`` 와
``torchvision`` 패키지가 없다면 설치합니다.

::

   pip install torch
   pip install torchvision


"""


######################################################################
# 단계(Steps)
# -----------
#
# 1단계부터 4단계까지는 학습을 위한 데이터와 신경망을 준비하며, 5단계에서 변화도를 0으로
# 만들어 줍니다. 이미 준비한 데이터와 신경망이 있다면 5단계로 건너뛰어도 좋습니다.
#
# 1. 데이터를 불러오기 위해 필요한 모든 라이브러리 import 하기
# 2. 데이터셋 불러오고 정규화하기
# 3. 신경망 구축하기
# 4. 손실 함수 정의하기
# 5. 신경망을 학습시킬 때 변화도 0으로 만들기
#
# 1. 데이터를 불러오기 위해 필요한 모든 라이브러리 import 하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 레시피에서는 데이터셋에 접근하기 위해 ``torch`` 와 ``torchvision`` 을 사용합니다.
#

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


######################################################################
# 2. 데이터셋 불러오고 정규화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch는 다양한 내장 데이터셋을 제공합니다. PyTorch에서 데이터 불러오기 레시피를 참고해
# 더 많은 정보를 얻을 수 있습니다.
#

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


######################################################################
# 3. 신경망 구축하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 컨볼루션 신경망을 정의하겠습니다. 자세한 내용은 신경망 정의하기 레시피를
# 참조해주세요.
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


######################################################################
# 4. 손실 함수과 옵티마이저 정의하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 분류를 위한 Cross-Entropy 손실 함수와 모멘텀을 설정한 SGD 옵티마이저를 사용합니다.
#

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 5. 신경망을 학습시키는 동안 변화도를 0으로 만들기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이제부터 흥미로운 부분을 살펴보려고 합니다.
# 여기서 할 일은 데이터 이터레이터를 순회하면서, 신경망에 입력을 주고
# 최적화하는 것입니다.
#
# 데이터의 엔터티 각각의 변화도를 0으로 만들어주는 것에 유의하십시오.
# 신경망을 학습시킬 때 불필요한 정보를 추적하지 않도록 하기 위함입니다.
#

for epoch in range(2):  # 전체 데이터셋을 여러번 반복하기

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 입력 받기: 데이터는 [inputs, labels] 형태의 리스트
        inputs, labels = data

        # 파라미터 변화도를 0으로 만들기
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.item()
        if i % 2000 == 1999:    # 미니배치 2000개 마다 출력
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


######################################################################
# ``model.zero_grad()`` 를 사용해도 변화도를 0으로 만들 수 있습니다.
# 이는 옵티마이저에 모든 모델 파라미터가 포함되는 한 ``optimizer.zero_grad()`` 를
# 사용하는 것과 동일합니다. 어떤 것을 사용할 것인지 최선의 선택을 하기 바랍니다.
#
# 축하합니다! 이제 PyTorch에서 변화도를 0으로 만들 수 있습니다.
#
# 더 알아보기
# -------------
#
# 다른 레시피를 둘러보고 계속 배워보세요:
#
# - :doc:`/recipes/recipes/loading_data_recipe`
# - :doc:`/recipes/recipes/save_load_across_devices`
