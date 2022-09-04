"""
PyTorch에서 일반적인 체크포인트(checkpoint) 저장하기 & 불러오기
===================================================================
추론(inference) 또는 학습(training)의 재개를 위해 체크포인트(checkpoint) 모델을
저장하고 불러오는 것은 마지막으로 중단했던 부분을 선택하는데 도움을 줄 수 있습니다.
체크포인트를 저장할 때는 단순히 모델의 state_dict 이상의 것을 저장해야 합니다.
모델 학습 중에 갱신되는 버퍼와 매개변수들을 포함하는 옵티마이저(Optimizer)의
state_dict를 함께 저장하는 것이 중요합니다. 이 외에도 중단 시점의 에포크(epoch),
마지막으로 기록된 학습 오차(training loss), 외부 ``torch.nn.Embedding`` 계층 등,
알고리즘에 따라 저장하고 싶은 항목들이 있을 것입니다.

개요
------------
여러 체크포인트들을 저장하기 위해서는 사전(dictionary)에 체크포인트들을 구성하고
``torch.save()`` 를 사용하여 사전을 직렬화(serialize)해야 합니다. 일반적인
PyTorch에서는 이러한 여러 체크포인트들을 저장할 때 ``.tar`` 확장자를 사용하는 것이
일반적인 규칙입니다. 항목들을 불러올 때에는, 먼저 모델과 옵티마이저를 초기화하고,
torch.load()를 사용하여 사전을 불러옵니다. 이후 원하는대로 저장한 항목들을 사전에
조회하여 접근할 수 있습니다.

이 레시피에서는 여러 체크포인트들을 어떻게 저장하고 불러오는지 살펴보겠습니다.

설정
-----
시작하기 전에 ``torch`` 가 없다면 설치해야 합니다.


::

   pip install torch


"""



######################################################################
# 단계(Steps)
# ------------
#
# 1. 데이터 불러올 때 필요한 라이브러리들 불러오기
# 2. 신경망을 구성하고 초기화하기
# 3. 옵티마이저 초기화하기
# 4. 일반적인 체크포인트 저장하기
# 5. 일반적인 체크포인트 불러오기
#
# 1. 데이터 불러올 때 필요한 라이브러리들 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 레시피에서는 ``torch`` 와 여기 포함된 ``torch.nn`` 와 ``torch.optim`` 을
# 사용하겠습니다.
#

import torch
import torch.nn as nn
import torch.optim as optim


######################################################################
# 2. 신경망을 구성하고 초기화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 예를 들어, 이미지를 학습하는 신경망을 만들어보겠습니다. 더 자세한 내용은
# `신경망 구성하기 레시피 <defining_a_neural_network.html>`_ 를 참고해주세요.
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
# 3. 옵티마이저 초기화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 모멘텀(momentum)을 갖는 SGD를 사용하겠습니다.
#

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 4. 일반적인 체크포인트 저장하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 관련된 모든 정보들을 모아서 사전을 구성합니다.
#

# 추가 정보
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)


######################################################################
# 5. 일반적인 체크포인트 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 먼저 모델과 옵티마이저를 초기화한 뒤, 사전을 불러오는 것을 기억하십시오.
#

model = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - 또는 -
model.train()


######################################################################
# 추론(inference)을 실행하기 전에 ``model.eval()`` 을 호출하여 드롭아웃(dropout)과
# 배치 정규화 층(batch normalization layer)을 평가(evaluation) 모드로 바꿔야한다는
# 것을 기억하세요. 이것을 빼먹으면 일관성 없는 추론 결과를 얻게 됩니다.
#
# 만약 학습을 계속하길 원한다면 ``model.train()`` 을 호출하여 이 층(layer)들이
# 학습 모드인지 확인(ensure)하세요.
#
# 축하합니다! 지금까지 PyTorch에서 추론 또는 학습 재개를 위해 일반적인 체크포인트를
# 저장하고 불러왔습니다.
#
# 더 알아보기
# ------------
#
# 다른 레시피를 둘러보고 계속 배워보세요:
#
# - :doc:`/recipes/recipes/saving_and_loading_a_general_checkpoint`
# - :doc:`/recipes/recipes/saving_multiple_models_in_one_file`
