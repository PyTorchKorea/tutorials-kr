# -*- coding: utf-8 -*-
"""
Ray Tune으로 하이퍼파라미터 튜닝하기
===================================

하이퍼파라미터 튜닝은 평균적인 모델과 매우 정확한 모델 간의 차이를 만들어냅니다.
학습률을 다르게 하거나 네트워크의 레이어 수를 바꾸는 간단한 방법만으로도 
종종 모델의 성능에 많은 영향을 끼칠 수 있습니다.

다행히도 가장 좋은 하이퍼파라미터의 조합을 찾는 데 도움을 주는 많은 툴들이 있는데요.
`Ray Tune <https://docs.ray.io/en/latest/tune.html>`은 분산 하이퍼파라미터 튜닝을 위한 
산업 표준 툴입니다. Ray Tune은 가장 최신의 하이퍼파라미터 검색 알고리즘을 포함하고 있으며
텐서보드 및 다른 분석 라이브러리들과 통합될 뿐만 아니라, `Ray의 분산 머신러닝 엔진<https://ray.io/>`을
통해 분산 학습을 기본적으로 지원합니다.

이번 튜토리얼에서는 Ray Tune을 파이토치의 학습 워크플로우에 통합하는 방법을 소개하고자 합니다.
파이토치 문서 중 `이 튜토리얼<https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html>`의 내용을 CIFAR10 이미지 분류기 학습에 확장시킬 것입니다.

확인하는 바와 같이 간단한 수정만 하면 됩니다. 구체적으로 아래의 작업들이 필요합니다.

1. 데이터 로딩과 학습하는 과정을 함수로 랩핑합니다. 
2. 네트워크의 매개변수가 설정 가능하도록 합니다.
3. 체크포인트를 추가합니다. (선택사항)
4. 모델 튜닝을 위한 탐색 공간을 정의합니다.

|

이 튜토리얼을 실행하기 위해서 다음의 패키지들이 설치되어있는지 확인해주세요.

-  ``ray[tune]``: 분산 하이퍼파라미터 튜닝 패키지
-  ``torchvision``: 데이터 트랜스포머를 위한 패키지

설치 / 패키지 불러오기
---------------
패키지를 불러오는 것으로 시작하겠습니다:
"""
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

######################################################################
# 라이브러리 대부분은 파이토치 모델을 빌딩하기 위해 필요한 것이고 
# 마지막 3개는 Ray Tune을 위한 것입니다. 
#
# 데이터 로더
# ------------
# 데이터 로더를 별도의 함수에 랩핑하고 전역 데이터 디렉토리를 전달합니다.
# 이렇게 함으로써, 다른 시도들 간에도 데이터 디렉토리를 공유할 수 있습니다.


def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

######################################################################
# 설정 가능한 신경망
# ---------------------------
# 설정 가능한 하이퍼 파라미터만 튜닝할 수 있습니다. 
# 이번 예제에서는 완전연결 신경망의 레이어 사이즈를 지정할 수 있습니다.


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

######################################################################
# 학습 함수
# ------------------
# 파이토치 문서의 예제`from the PyTorch ocumentation <https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html>`에 변화를 줄 것이기 때문에.
# 이제부터 본격적으로 재미있어질 거예요.
#
# 학습 스크립트를 ``train_cifar(config, checkpoint_dir=None, data_dir=None)``라는 함수로 랩핑합니다.
# 예상할 수 있는 것처럼 ``config`` 파라미터는 parameter 우리가 학습에 사용하고자 하는 하이퍼파라미터를 받습니다.
# ``checkpoint_dir`` 파라미터는 체크포인트를 복구하기 위해 사용됩니다.
# ``data_dir`` 는 여러번의 시도가 동일한 데이터 소스를 공유할 수 있도록 데이터를 로드하고 저장하는 경로를 지정합니다. 
#
# .. code-block:: python
#
#     net = Net(config["l1"], config["l2"])
#
#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(
#             os.path.join(checkpoint_dir, "checkpoint"))
#         net.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)
#
# 최적화방식의 학습률도 설정 가능하도록 만들어줍니다.
#
# .. code-block:: python
#
#     optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
#
# 학습 데이터를 학습 데이터와 검증 데이터로 한 번 더 분할합니다. 
# 따라서 학습 데이터의 80%에 대해 학습을 진행하고 남은 20%에 대해서는 검증 손실을 계산하게 됩니다.
# 학습 데이터셋과 테스트 데이터셋을 반복할 때의 배치 사이즈 또한 설정 가능합니다.
#
# DataParallel로 (멀티) GPU 지원 추가하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GPU를 사용하면 이미지 분류에 많은 이점이 있습니다 .다행히도 Ray Tune에서도 파이토치의 추상화도 계속해서
# 사용할 수 있습니다. 따라서 ``nn.DataParallel``에 여러 개의 GPU상에서의 데이터 병렬 학습을 지원하기 위해 모델을 랩핑할 수 있습니다.:
#
# .. code-block:: python
#
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             net = nn.DataParallel(net)
#     net.to(device
#
# ``device`` 변수를 사용함으로써 GPU가 사용 가능하지 않을 떄에느 학습이 잘 이루어질 수 있도록 합니다. 
# 파이토치는 아래와 같이 데이터를 GPU 메모리에 명시적으로 올리도록 요구합니다. 
# like this:
#
# .. code-block:: python
#
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#
# 이 코드는 이제 CPU와 단일 GPU, 여러개의 GPU 상에서의 학습을 지원합니다. 
# 특히 Ray는 `fractional GPUs <https://docs.ray.io/en/master/using-ray-with-gpus.html#fractional-gpus>`을 지원하여
#  모델이 GPU 메모리에 적합한 한, 여러 번의 시도 간에 GPU를 공유할 수 있도록 합니다.
#  이 부분에 대해서는 나중에 다시 돌아오도록 하겠습니다.
#
# Ray Tune과의 커뮤니케이션
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 가장 흥미로운 부분은 Ray Tune과의 커뮤니케이션입니다:
#
# .. code-block:: python
#
#     with tune.checkpoint_dir(epoch) as checkpoint_dir:
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         torch.save((net.state_dict(), optimizer.state_dict()), path)
#
#     tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
#
# 여기서 우리는 먼저 체크포인트를 저장하고 Ray Tune에게 어떤 메트릭을 리포트합니다. 
# 구체적으로 검증 손실과 정확도를 Ray Tune에게 보냅니다. 그러면 Ray Tune은 이 메트릭을 이용해서 
# 어떤 하이퍼파라미터 설정이 가장 좋은 결과를 낼 수 있을지 결정하게 됩니다.
# 이 메트릭은 또한 성능이 안 좋은 시도를 조기에 중단시켜 해당 시도에 대해 리소스가 낭비되는 것을 방지합니다.
#
# 체크포인트 저장은 선택사항이지만 우리가 `Population Based Training <https://docs.ray.io/en/master/tune/tutorials/tune-advanced-tutorial.html>`_.
# 와 같은 고급 스케줄러를 사용하기 위해서는 필수적입니다. 
# 또한 체크포인트를 저장함으로써 학습된 모델을 나중에 로드하고 테스트 데이터셋에 대해검증할 수 있습니다.
#
# 전체 학습 함수
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 전체 예제 코드는 이와 같습니다:


def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # 데이터셋에 대해 여러번 루프문이 실행됩니다.
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # 입력을 받습니다; 데이터는 [inputs, labels]의 리스트입니다.
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 파라미터 그레디언트를 0으로 설정합니다.
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계량을 출력합니다.
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # 2000개의 모든 미니배치를 출력합니다
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # 검증 손실(validation loss)
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

######################################################################
# 보시다시피, 이 코드의 대부분은 원래의 예제에서 직접적으로 수정되었습니다.
#
# 테스트셋 정확도
# -----------------
# 일반적으로 머신러닝 모델의 성능은 모델을 학습하는데 사용되지 않은 데이터로 이루어진 홀드아웃 테스트 데이터 셋에 대해 평가가 이루어집니다. 
# 이 또한 함수로 랩핑합니다. 


def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

######################################################################
# 이 함수도 GPU 상에서 테스트 셋 검증을 할 수 있도록``device`` 파라미터를 필요로 합니다.
#
# 탐색 공간 설정
# ----------------------------
# 마지막으로 Ray Tune의 탐색 공간을 정의해야 합니다. 여기 예제를 보세요:
#
# .. code-block:: python
#
#     config = {
#         "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
#
# ``tune.sample_from()`` 함수는 하이퍼파라미터를 구하기 위한 사용자의 샘플 방법을 정의하도록 해줍니다.
# 이 예시에서 ``l1``과 ``l2`` 파라미터는 4, 8, 16, 32, 64, 128 또는 256과 같이 
# 4와 256 사이의 2의 거듭제곱이어야 합니다. ``lr`` (학습률)은 0.0001과 0.1 사이에서 균등하게 추출되어야 합니다.
# 마지막으로 배치사이즈는 2, 4, 8, 그리고 16 중에 선택되어야 합니다.
#
# 각 시도에서 Ray Tune은 이제 이 탐색 공간으로부터 하이퍼파라미터의 조합을 랜덤하게 추출합니다. 
# 여러 모델을 병렬적으로 학습하고 그 중에서 가장 좋은 성능을 내는 모델을 찾게 됩니다.
# ``ASHAScheduler``라는 스케줄러를 사용하는데 성능이 안좋은 시도는 조기에 종료되도록 합니다.
#
# ``data_dir`` 파라미터를 고정시키기 위해 ``train_cifar`` function with ``functools.partial``를 랩핑할 것입니다.
# 또한 각 시도에서 리소스가 어떻게 사용되는지 Ray Tune에게 알려줄 수도 있습니다:
#
# .. code-block:: python
#
#     gpus_per_trial = 2
#     # ...
#     result = tune.run(
#         partial(train_cifar, data_dir=data_dir),
#         resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         checkpoint_at_end=True)
#
# 사용 가능한 CPU의 수를 지정할 수도 있습니다. 에를 들어, 파이토치 ``DataLoader`` 인스턴스의 ``num_workers``를 늘릴 수도 있습니다.
# 선택된 GPU수는 각 시도에서 파이토치에 보여집니다.  
# 각 시도들은 요청받지 않은 GPU에 대해서는 접근할 수 없기 때문에 동일한 리소스를 사용하는 두 시도에 대해서도 신경쓰지 않을 수 있습니다.
#
# 여기서 GPU 개수를 소수로도 지정할 수 있기 때문에, ``gpus_per_trial=0.5``같은 것도 완전히 유효합니다.
# 각 시도들은 서로 간에 GPU를 공유하게 됩니다. 여러분은 단지 모델이 GPU 메모리에 적합한지 확인해주면 됩니다.
#
# 모델을 학습한 이후에, 가장 성능이 우수한 모델을 찾고 체크포인트 파일에서 
# 학습된 네트워크를 로드합니다. 그리고 테스트 셋 정확도를 구하고 결과를 출력하여 리포트합니다.
#
# 전체 메인 함수는 이와 같이 구성됩니다.


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # 여기서 각 시도마다 GPU의 수를 바꿀 수 있습니다.
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)


######################################################################
# 이 코드를 실행하면, 결과 예시는 아래와 같은 형태를 보입니다.
#
# ::
#
#     Number of trials: 10 (10 TERMINATED)
#     +-----+------+------+-------------+--------------+---------+------------+--------------------+
#     | ... |   l1 |   l2 |          lr |   batch_size |    loss |   accuracy | training_iteration |
#     |-----+------+------+-------------+--------------+---------+------------+--------------------|
#     | ... |   64 |    4 | 0.00011629  |            2 | 1.87273 |     0.244  |                  2 |
#     | ... |   32 |   64 | 0.000339763 |            8 | 1.23603 |     0.567  |                  8 |
#     | ... |    8 |   16 | 0.00276249  |           16 | 1.1815  |     0.5836 |                 10 |
#     | ... |    4 |   64 | 0.000648721 |            4 | 1.31131 |     0.5224 |                  8 |
#     | ... |   32 |   16 | 0.000340753 |            8 | 1.26454 |     0.5444 |                  8 |
#     | ... |    8 |    4 | 0.000699775 |            8 | 1.99594 |     0.1983 |                  2 |
#     | ... |  256 |    8 | 0.0839654   |           16 | 2.3119  |     0.0993 |                  1 |
#     | ... |   16 |  128 | 0.0758154   |           16 | 2.33575 |     0.1327 |                  1 |
#     | ... |   16 |    8 | 0.0763312   |           16 | 2.31129 |     0.1042 |                  4 |
#     | ... |  128 |   16 | 0.000124903 |            4 | 2.26917 |     0.1945 |                  1 |
#     +-----+------+------+-------------+--------------+---------+------------+--------------------+
#
#
#     Best trial config: {'l1': 8, 'l2': 16, 'lr': 0.00276249, 'batch_size': 16, 'data_dir': '...'}
#     Best trial final validation loss: 1.181501
#     Best trial final validation accuracy: 0.5836
#     Best trial test set accuracy: 0.5806
#
# 대부분의 시도들이 리소스 낭비를 피하기 위해 조기 종료되었습니다.
# 가장 높은 성능의 시도는 약 58%의 검증 정확도를 달성하였으며, 테스트 셋에서도 이를 확인할 수 있습니다.
#
# 이번 튜토리얼은 여기까지입니다. 이제 우리가 가진 파이토치 모델의 하이퍼파라미터를 직접 튜닝할 수 있습니다.
