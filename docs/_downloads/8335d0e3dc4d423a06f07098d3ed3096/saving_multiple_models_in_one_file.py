"""
PyTorch에서 여러 모델을 하나의 파일에 저장하기 & 불러오기
============================================================
여러 모델을 저장하고 불러오는 것은 이전에 학습했던 모델들을 재사용하는데 도움이 됩니다.

개요
------------
GAN이나 시퀀스-투-시퀀스(sequence-to-sequence model), 앙상블 모델(ensemble of models)과
같이 여러 ``torch.nn.Modules`` 로 구성된 모델을 저장할 때는 각 모델의 state_dict와
해당 옵티마이저(optimizer)의 사전을 저장해야 합니다. 또한, 학습 학습을 재개하는데
필요한 다른 항목들을 사전에 추가할 수 있습니다. 모델들을 불러올 때에는, 먼저
모델들과 옵티마이저를 초기화하고, ``torch.load()`` 를 사용하여 사전을 불러옵니다.
이후 원하는대로 저장한 항목들을 사전에 조회하여 접근할 수 있습니다.
이 레시피에서는 PyTorch를 사용하여 여러 모델들을 하나의 파일에 어떻게 저장하고
불러오는지 살펴보겠습니다.

설정
---------
시작하기 전에 ``torch`` 가 없다면 설치해야 합니다.

::

   pip install torch

"""



######################################################################
# 단계(Steps)
# -------------
#
# 1. 데이터 불러올 때 필요한 라이브러리들 불러오기
# 2. 신경망을 구성하고 초기화하기
# 3. 옵티마이저 초기화하기
# 4. 여러 모델들 저장하기
# 5. 여러 모델들 불러오기
#
# 1. 데이터 불러올 때 필요한 라이브러리들 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 레시피에서는 ``torch`` 와 여기 포함된 ``torch.nn`` 와 ``torch.optim` 을
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
# 신경망 구성하기 레시피를 참고해주세요. 모델을 저장할 2개의 변수들을 만듭니다.
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

netA = Net()
netB = Net()


######################################################################
# 3. 옵티마이저 초기화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 생성한 모델들 각각에 모멘텀(momentum)을 갖는 SGD를 사용하겠습니다.
#

optimizerA = optim.SGD(netA.parameters(), lr=0.001, momentum=0.9)
optimizerB = optim.SGD(netB.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 4. 여러 모델들 저장하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 관련된 모든 정보들을 모아서 사전을 구성합니다.
#

# 저장할 경로 지정
PATH = "model.pt"

torch.save({
            'modelA_state_dict': netA.state_dict(),
            'modelB_state_dict': netB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            }, PATH)


######################################################################
# 5. 여러 모델들 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 먼저 모델과 옵티마이저를 초기화한 뒤, 사전을 불러오는 것을 기억하십시오.
#

modelA = Net()
modelB = Net()
optimModelA = optim.SGD(modelA.parameters(), lr=0.001, momentum=0.9)
optimModelB = optim.SGD(modelB.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - 또는 -
modelA.train()
modelB.train()


######################################################################
# 추론(inference)을 실행하기 전에 ``model.eval()`` 을 호출하여 드롭아웃(dropout)과
# 배치 정규화 층(batch normalization layer)을 평가(evaluation) 모드로 바꿔야한다는
# 것을 기억하세요. 이것을 빼먹으면 일관성 없는 추론 결과를 얻게 됩니다.
#
# 만약 학습을 계속하길 원한다면 ``model.train()`` 을 호출하여 이 층(layer)들이
# 학습 모드인지 확인(ensure)하세요.
#
# 축하합니다! 지금까지 PyTorch에서 여러 모델들을 저장하고 불러왔습니다.
#
# 더 알아보기
# ------------
#
# 다른 레시피를 둘러보고 계속 배워보세요:
#
# - :doc:`/recipes/recipes/saving_and_loading_a_general_checkpoint`
# - :doc:`/recipes/recipes/saving_multiple_models_in_one_file`
#
