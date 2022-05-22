"""
PyTorch로 TensorBoard 사용하기
===================================
TensorBoard는 머신러닝 실험을 위한 시각화 툴킷(toolkit)입니다.
TensorBoard를 사용하면 손실 및 정확도와 같은 측정 항목을 추적 및 시각화하는 것,
모델 그래프를 시각화하는 것, 히스토그램을 보는 것, 이미지를 출력하는 것 등이 가능합니다.
이 튜토리얼에서는 TensorBoard 설치, PyTorch의 기본 사용법,
TensorBoard UI에 기록한 데이터를 시각화 하는 방법을 다룰 것입니다.

설치하기
----------------------
모델과 측정 항목을 TensorBoard 로그 디렉터리에 기록하려면 PyTorch를 설치해야 합니다.
Anaconda를 통해 PyTorch 1.4 이상을 설치하는 방법은 다음과 같습니다.(권장):
::

   $ conda install pytorch torchvision -c pytorch


또는 pip를 사용할 수도 있습니다.

::

   $ pip install torch torchvision

"""

######################################################################
# PyTorch로 TensorBoard 사용하기
# -----
#
# 이제 PyTorch로 TensorBoard를 사용해봅시다!
# 먼저 ``SummaryWriter`` 인스턴스를 생성해야 합니다.
#

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

######################################################################
# Writer는 기본적으로 ``./runs/`` 디렉터리에 출력됩니다.
#


######################################################################
# 스칼라(scalar) 기록하기
# -------------------------
#
# 머신러닝에서는 손실 같은 주요 측정 항목과 학습 중 그것이 어떻게 변하는지 이해하는 것이
# 중요합니다. 스칼라는 각 학습 단계(step)에서의 손실 값이나 각 에폭 이후의 정확도를 저장하는 데
# 도움을 줍니다.
#
# 스칼라 값을 기록하려면 ``add_scalar(tag, scalar_value, global_step=None, walltime=None)``
# 을 사용해야 합니다. 예로, 간단한 선형 회귀 학습을 만들고 ``add_scalar`` 를 사용해
# 손실 값을 기록해 봅시다.

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(10)
writer.flush()


######################################################################
# 모든 보류중인(pending) 이벤트가 디스크에 기록되었는지 확인하려면 ``flush()``
# 메소드를 호출합니다.
#
# 기록할 수 있는 더 많은 TensorBoard 시각화 방법을 찾으려면
# `torch.utils.tensorboard tutorials <https://pytorch.org/docs/stable/tensorboard.html>`_ 을
# 참조하세요.
#
# Summary writer가 더 이상 필요하지 않으면 ``close()`` 메소드를 호출합니다.
#

writer.close()

######################################################################
# TensorBoard 실행하기
# -----
#
# 기록한 데이터를 시각화하기 위해서 다음과 같이 TensorBoard를 설치합니다.
#
# ::
#
#    $ pip install tensorboard
#
#
# 이제, 위에서 사용한 루트 로그 디렉터리를 지정하여 TensorBoard를 시작합니다.
# ``logdir`` 인자는 TensorBoard가 출력할 수 있는 이벤트 파일들을 찾을 디렉터리를 가리킵니다.
# TensorBoard는 .*tfevents.* 파일을 찾기 위해 logdir의 디렉터리 구조를 재귀적으로 탐색합니다.
#
# ::
#
#    $ tensorboard --logdir=runs
#
# 제공하는 URL로 이동하거나 `http://localhost:6006/ <http://localhost:6006/>`_ 로 이동합니다.
#
# .. image:: ../../_static/img/thumbnails/tensorboard_scalars.png
#    :scale: 40 %
#
# 이 대시보드는 매 에폭마다 손실과 정확도가 어떻게 변하는지 보여줍니다.
# 이를 사용하여 학습 속도, 학습률 및 기타 스칼라 값들을 추적할 수도 있습니다.
# 모델을 향상시키려면 여러 다른 학습을 돌리면서 이러한 측정 기준들을 비교하는 것이 좋습니다.


######################################################################
# TensorBoard 대시보드 공유하기
# -----
#
# `TensorBoard.dev <https://tensorboard.dev/>`_ 를 사용해 ML 실험 결과를
# 업로드하고 모두와 공유할 수 있습니다. TensorBoard.dev를 사용하여
# TensorBoard 대시보드를 호스팅, 추적 및 공유하세요.
#
# 업로더(uploader)를 사용하려면 TensorBoard 최신 버전을 설치하세요.
#
# ::
#
#    $ pip install tensorboard --upgrade
#
# 다음과 같은 명령을 사용하여 TensorBoard를 업로드하고 공유하세요.
#
# ::
#
#   $ tensorboard dev upload --logdir runs \
#   --name "My latest experiment" \ # 선택 사항
#   --description "Simple comparison of several hyperparameters" # 선택 사항
#
# 도움이 필요하면 ``$ tensorboard dev --help`` 를 실행하세요.
#
# **참고:** 업로드한 TensorBoard는 공개되어 누구나 볼 수 있게 됩니다.
# 민감한 데이터가 있다면 업로드하지 마세요.
#
# 터미널에서 제공한 URL로 TensorBoard를 실시간으로 확인하세요.
# 예: `https://tensorboard.dev/experiment/AdYd1TgeTlaLWXx6I8JUbA <https://tensorboard.dev/experiment/AdYd1TgeTlaLWXx6I8JUbA>`_
#
#
# .. image:: ../../_static/img/thumbnails/tensorboard_dev.png
#    :scale: 40 %
#
#
# .. note::
#   TensorBoard.dev는 현재 스칼라(scalar), 그래프(graph), 히스토그램(historgram), 분포(distribution), hparam과 텍스트(text) 대시보드들을 지원합니다.

########################################################################
# 더 알아보기
# ----------------------------
#
# -  `torch.utils.tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ docs
# - :doc:`/intermediate/tensorboard_tutorial` 튜토리얼
#
