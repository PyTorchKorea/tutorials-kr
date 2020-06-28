"""
PyTorch에서 추론(inference)을 위해 모델 저장하기 & 불러오기
================================================================

PyTorch에서는 추론(inference)을 위해 모델을 저장하고 불러오는데 2가지 접근법이
있습니다. 첫번째는 ``state_dict`` 를 저장하고 불러오는 것이고, 두번째는 전체
모델을 저장하는 것입니다.

개요
------------
torch.save() 함수를 사용하여 모델의 ``state_dict`` 를 저장하면 이후에 모델을
불러올 때 유연함을 크게 살릴 수 있습니다. 학습된 모델의 매개변수(parameter)만을
저장하면되므로 모델 저장 시에 권장하는 방법입니다. 모델 전체를 저장하고 불러올
때에는 Python의 `pickle <https://docs.python.org/3/library/pickle.html>`__ 모듈을
사용하여 전체 모듈을 저장합니다. 이 방식은 직관적인 문법을 사용하며 코드의 양도
적습니다. 이 방식의 단점은 직렬화(serialized)된 데이터가 모델을 저장할 때 사용한
특정 클래스 및 디렉토리 구조에 종속(bind)된다는 것입니다. 그 이유는 pickle이
모델 클래스 자체를 저장하지 않기 때문입니다. 오히려 불러올 때 사용되는 클래스가
포함된 파일의 경로를 저장합니다. 이 때문에 작성한 코드가 다른 프로젝트에서
사용되거나 리팩토링을 거치는 등의 과정에서 동작하지 않을 수 있습니다. 이 레시피에서는
추론을 위해 모델을 저장하고 불러오는 두 가지 방법 모두를 살펴보겠습니다.

설정
----------
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
# 4. ``state_dict`` 을 통해 모델을 저장하고 불러오기
# 5. 전체 모델을 저장하고 불러오기
#
# 1. 데이터 불러올 때 필요한 라이브러리들 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 레시피에서는 ``torch`` 와 여기 포함된 ``torch.nn`` 과 ``torch.optim` 을
# 사용하겠습니다.
#

import torch
import torch.nn as nn
import torch.optim as optim


######################################################################
# 2. 신경망을 구성하고 초기화하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 예를 들기 위해, 이미지를 학습하는 신경망을 만들어보겠습니다. 더 자세한 내용은
# 신경망 구성하기 레시피를 참고해주세요.
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
# 4. ``state_dict`` 을 통해 모델을 저장하고 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이제 ``state_dict`` 만 사용하여 모델을 저장하고 불러와보겠습니다.
#

# 경로 지정
PATH = "state_dict_model.pt"

# 저장하기
torch.save(net.state_dict(), PATH)

# 불러오기
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()


######################################################################
# PyTorch에서는 모델을 저장할 때 ``.pt`` 또는 ``.pth`` 확장자를 사용하는 것이
# 일반적인 규칙입니다.
#
# ``load_state_dict()`` 함수는 저장된 객체의 경로가 아닌, 사전 객체를 사용합니다.
# 즉, 저장된 state_dict를 ``load_state_dict()`` 함수에 전달하기 전에 반드시
# 역직렬화(deserialize)를 해야 합니다. 예를 들어, ``model.load_state_dict(PATH)``
# 와 같이 사용할 수 없습니다.
#
# 또한, 추론을 실행하기 전에 ``model.eval()`` 을 호출하여 드롭아웃(dropout)과
# 배치 정규화 층(batch normalization layers)을 평가(evaluation) 모드로 바꿔야한다는
# 것을 기억하세요. 이것을 빼먹으면 일관성 없는 추론 결과를 얻게 됩니다.
#
# 5. 전체 모델을 저장하고 불러오기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이제 전체 모델에 대해서 똑같이 해보겠습니다.
#

# 경로 지정
PATH = "entire_model.pt"

# 저장하기
torch.save(net, PATH)

# 불러오기
model = torch.load(PATH)
model.eval()


######################################################################
# 여기서도 또한 model.eval()을 실행하여 드롭아웃(dropout)과 배치 정규화 층
# (batch normalization layers)을 평가(evaluation) 모드로 바꿔야한다는
# 것을 기억하세요.
#
# 축하합니다! 지금까지 PyTorch에서 추론을위한 모델을 성공적으로 저장하고 불러왔습니다.
#
# 더 알아보기
# -------------
#
# 다른 레시피를 둘러보고 계속 배워보세요:
#
# - :doc:`/recipes/recipes/saving_and_loading_a_general_checkpoint`
# - :doc:`/recipes/recipes/saving_multiple_models_in_one_file`
