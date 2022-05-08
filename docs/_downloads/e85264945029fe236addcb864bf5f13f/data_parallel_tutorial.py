"""
선택 사항: 데이터 병렬 처리 (Data Parallelism)
====================================================

**글쓴이**: `Sung Kim <https://github.com/hunkim>`_ and `Jenny Kang <https://github.com/jennykang>`_
**번역**: '정아진 <https://github.com/ajin-jng>'

이 튜토리얼에서는 ``DataParallel`` (데이터 병렬) 을 사용하여 여러 GPU를 사용하는 법을 배우겠습니다.

PyTorch를 통해 GPU를 사용하는 것은 매우 쉽습니다. 먼저, 모델을 GPU에 넣습니다:

.. code:: python

    device = torch.device("cuda:0")
    model.to(device)

그 다음으로는 모든 Tensors 를 GPU로 복사합니다:

.. code:: python

    mytensor = my_tensor.to(device)

''my_tensor.to(device)'' 를 호출 시 에는 ''my_tensor'' 를 다시쓰는 대신 ''my_tensor'' 의 또다른 복사본이 생긴다는 사실을 기억하십시오.
당신은 그것을 새로운 tensor 에 소속시키고 GPU에 그 tensor를 써야합니다.

여러 GPU를 통해 앞과 뒤의 전파를 실행하는 것은 당연한 일 입니다.
그러나 PyTorch는 기본적 하나의 GPU만 사용합니다. ``DataParallel`` 을 이용하여 모델을 병렬로 실행하여 다수의 GPU 에서 쉽게 작업을 실행할 수 있습니다:

.. code:: python

    model = nn.DataParallel(model)

이것이 튜토리얼의 핵심입니다. 자세한 사항들에 대해서는 아래에서 살펴보겠습니다.
"""


######################################################################
# 불러오기와 매개변수
# ----------------------
#
# PyTorch 모듈을 불러오고 매개변수를 정의하십시오.
#

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 매개변수와 DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100


######################################################################
# 기기
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# 더미(Dummy) 데이터셋
# -----------------------
#
# 더미(ramdom) 데이터셋을 만들어 봅시다. Getitem 만 구현하면 됩니다.
#

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


######################################################################
# 간단한 모델
# ------------
#
# 데모를 위해 모델은 입력을 받고 선형 연산을 수행하며 출력을 제공합니다. 그러나 ``DataParallel`` 의 어떤 모델 (CNN, RNN, Capsule Net 등) 에서든 사용할 수 있습니다.
#
# 우리는 input과 output의 크기를 모니터링하기 위해 모델안에 print 문을 넣었습니다.
# 무엇이 배치 순위 (batch rank) 0 에 프린트 되는지 주의 깊게 봐주시길 바랍니다.
#

class Model(nn.Module):
    # 우리의 모델

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


######################################################################
# 모델과 데이터 병렬의 구현
# -----------------------------
#
# 이것은 이 튜토리얼의 핵심 부분입니다. 먼저, model instance 를 만들고 가지고 있는 GPU가 여러개인지 확인해야합니다.
# 만약 다수의 GPU를 보유중이라면, ``nn.DataParallel`` 을 사용하여 모델을 래핑 (wrapping) 할 수 있습니다.
# 그런 다음 ``model.to(device)`` 를 통하여 모델을 GPU에 넣을 수 있습니다.
#

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)


######################################################################
# 모델 실행
# -------------
#
# 이제 우리는 입력과 출력 tensor의 크기를 볼 수 있습니다
#

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())


######################################################################
# 결과
# -------
#
# GPU가 없거나 하나인 경우 30개의 입력과 30개의 출력을 일괄 처리하면 모델이 예상대로 30을 입력받고 30을 출력합니다.
# 하지만 만약 당신이 다수의 GPU를 가지고 있다면, 다음과 같은 결과를 얻을 수 있습니다.
#
# 2 GPUs
# ~~~~~~
#
# 2개의 GPU를 갖고 있다면, 다음과 같이 표시될 것 입니다:
#
# .. code:: bash
#
#     # on 2 GPUs
#     Let's use 2 GPUs!
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#         In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#
# 3 GPUs
# ~~~~~~
#
# 만약 당신이 3개의 GPU를 갖고 있다면, 다음과 같이 표시될 것 입니다:
# .. code:: bash
#
#     Let's use 3 GPUs!
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#
# 8 GPUs
# ~~~~~~~~~~~~~~
#
# 만약 당신이 8개의 GPU를 갖고 있다면, 다음과 같이 표시될 것 입니다:
#
# .. code:: bash
#
#     Let's use 8 GPUs!
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#


######################################################################
# 요약
# -------
#
# DataParallel은 당신의 데이터를 자동으로 분할하고 여러 GPU에 있는 다수의 모델에 작업을 지시합니다. 각 모델이 작업을 완료하면 DataParallel은
# 사용자에게 결과를 반환하기 전에 모든 결과물들을 수집하여 병합합니다.
#
# 더 많은 정보를 알고싶다면 아래 주소를 참고하세요.
# https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
#
