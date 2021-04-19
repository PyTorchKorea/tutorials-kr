"""
PyTorch에서 state_dict란 무엇인가요?
======================================
PyTorch에서 ``torch.nn.Module`` 모델의 학습 가능한
매개변수(예. 가중치와 편향)들은 모델의 매개변수에 포함되어 있습니다.
(model.parameters()로 접근합니다)
``state_dict`` 는 간단히 말해 각 계층을 매개변수 텐서로 매핑되는
Python 사전(dict) 객체입니다.

개요
------------
``state_dict`` 는 PyTorch에서 모델을 저장하거나 불러오는 데 관심이
있다면 필수적인 항목입니다.
``state_dict`` 객체는 Python 사전이기 때문에 쉽게 저장, 업데이트,
변경 및 복원할 수 있으며, 이는 PyTorch 모델과 옵티마이저에 엄청난
모듈성(modularity)을 제공합니다.
이 때, 학습 가능한 매개변수를 갖는 계층(합성곱 계층, 선형 계층 등)
및 등록된 버퍼들(batchnorm의 running_mean)만 모델의 ``state_dict``
 항목을 가진다는 점에 유의하시기 바랍니다. 옵티마이저 객체
( ``torch.optim`` ) 또한 옵티마이저의 상태 뿐만 아니라 사용된
하이퍼 매개변수 (Hyperparameter) 정보가 포함된 ``state_dict`` 을
갖습니다.
레시피에서 ``state_dict`` 이 간단한 모델에서 어떻게 사용되는지
살펴보겠습니다.

설정
----------
시작하기 전에 ``torch`` 가 없다면 설치해야 합니다.

::

   pip install torch

"""



######################################################################
# 단계(Steps)
# --------------
#
# 1. 데이터를 불러올 때 필요한 모든 라이브러리 불러오기
# 2. 신경망을 구성하고 초기화하기
# 3. 옵티마이저 초기화하기
# 4. 모델과 옵티마이저의 ``state_dict`` 접근하기
#
# 1. 데이터를 불러올 때 필요한 모든 라이브러리 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 레시피에서는 ``torch`` 와 하위 패키지인 ``torch.nn`` 과 ``torch.optim`` 을
# 사용하겠습니다.
#

import torch
import torch.nn as nn
import torch.optim as optim


######################################################################
# 2. 신경망을 구성하고 초기화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 예시를 보이기 위해, 이미지를 학습하는 신경망을 만들어보겠습니다.
# 더 자세한 내용은 신경망 구성하기 레시피를 참고해주세요.
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
# 4. 모델과 옵티마이저의 ``state_dict`` 접근하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이제 모델과 옵티마이저를 구성했으므로 각각의 ``state_dict`` 속성에
# 저장되어 있는 항목을 확인할 수 있습니다.
#

# 모델의 state_dict 출력
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print()

# 옵티마이저의 state_dict 출력
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


######################################################################
# 이 정보는 향후 모델 및 옵티마이저를 저장하고
# 불러오는 것과 관련이 있습니다.
#
# 축하합니다! PyTorch에서 ``state_dict`` 을 성공적으로 사용하였습니다.
#
# 더 알아보기
# -------------
#
# 다른 레시피를 둘러보고 계속 배워보세요:
#
# - :doc:`/recipes/recipes/saving_and_loading_models_for_inference`
# - :doc:`/recipes/recipes/saving_and_loading_a_general_checkpoint`
