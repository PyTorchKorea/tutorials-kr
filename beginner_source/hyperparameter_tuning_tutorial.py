# -*- coding: utf-8 -*-
"""
RAY TUNE을 사용한 하이퍼파라미터 튜닝
===================================

하이퍼파라미터 튜닝을 통해서 평균적인 모델보다 더 정확한 모델을 만들 수 있습니다.
간혹 다른 학습률(learning rate)을 적용하거나 네트워크 계층의 크기(network layer size)를 
바꾸는 것과 같은 간단한 행동들이 모델 성능에 극적인 영향을 미칠 수 있습니다.

다행히도, 파라미터들의 최적의 조합을 찾을 때 도움이 되는 수단들이 있습니다.
`Ray Tune <https://docs.ray.io/en/latest/tune.html>`_은 분산(distributed) 하이퍼파라미터 튜닝을 위한 업계표준 수단입니다.
Ray Tune은 최신 하이퍼파라미터 탐색(search) 알고리즘을 포함하고,
텐서보드(TensorBoard)와 그 밖의 분석 라이브러리들과 통합되며, 또한
`Ray의 분산기계학습 엔진 <https://ray.io/>`_을 통해 분산훈련(distributed training)을 기본적으로 지원합니다.

이 튜토리얼을 통해서 Ray Tune을 PyTorch 훈련 워크플로우에 통합하는 방법을 보여드리겠습니다.
`해당 PyTorch 문서(documentation) <https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html>`_에서
확장해 CIFAR10 이미지 분류기(classifier)를 훈련시킬것입니다.

보시다시피, 몇가지 수정만 추가하면 됩니다.

1. 데이터 불러오기 그리고 훈련(training)을 각각의 함수들로 포장(wrap)해 줍니다,
2. 변경 가능한 네트워크 파라미터들을 구성합니다,
3. 체크포인트를 추가합니다 (선택사항),
4. 모델 튜닝을 위한 탐색범위(search space)를 정의합니다

|

이 튜토리얼을 실행하기 위해서, 아래와 같은 패키지들이 설치되어야합니다:

-  ``ray[tune]``: 분산 하이퍼파라미터 튜닝 라이브러리
-  ``torchvision``: 데이터 변환용(transformers)

설정/가져오기(Imports)
---------------
imports를 시작으로 진행합니다:
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
# 대부분의 imports는 PyTorch 모델을 구성하기위해 필요한 것들입니다. 
# 마지막 세개의 imports만 Ray Tune을 위한것입니다.
#
# 데이터 로더(loaders)
# ------------
# 데이터 로더(loaders)를 함수로 포장(wrap)하고 전역(global) 데이터 디렉토리(directory)를 전달합니다.
# 이 방법을 통해 서로다른 시도(trials)간 데이터 디렉토리(directory)를 공유할 수 있습니다.


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
# 변경 가능한 인공신경망(neural network)
# ---------------------------
# 변경 가능한 파라미터들만 조정(tune)할 수 있습니다. 이 예시에서는 
# 완전연결계층(fully connected layers)에서 계층(layer)의 크기가 해당됩니다.


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
# 훈련 함수(The train function)
# ------------------
# `이 PyTorch documentation <https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html>`_에서 
# 수정할 부분을 소개하겠습니다.
#
# 훈련 스크립트(script)를 함수로 포장(wrap)합니다 ``train_cifar(config, checkpoint_dir=None, data_dir=None)``.
# 짐작과 같이 ``config`` 파라미터는 훈련하고자 하는 하이퍼파라미터들을 전달받게됩니다.
# ``checkpoint_dir`` 파라미터는 체크포인트로 복원할 때 사용됩니다.
# ``data_dir`` 가 데이터를 불러오고 저장할 디렉토리를 지정함으로써 다중 실행시에도 동일한 데이터 소스를 공유할 수 있습니다.
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
# optimizer의 학습률(learning rate) 또한 변경할 수 있습니다.
#
# .. code-block:: python
#
#     optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
#
# 또한 훈련(training) 데이터를 훈련 데이터와 검증(validation) 데이터로 나눌 수 있습니다.
# 따라서 데이터의 80%를 훈련에 사용하고 남은 20%로 검증데이터에 대한 손실(the validation loss)을 계산할 수 있습니다.
# 훈련 및 테스트 셋을 통해 반복되는 배치 크기(batch size)도 변경할 수 있습니다.
#
# DataParallel을 통한 (다중) GPU 지원 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 이미지 분류는 GPU에서 얻는 이점이 매우 많습니다.
# 다행히도 Ray Tune 에서 PyTorch의 추상화 개념(abstractions)을 계속 사용할 수 있습니다.
# 따라서 다중(multiple) GPU에서의 데이터 병렬(parallel) 훈련(training)을 위해 모델을 nn.DataParallel 내에 포장(wrap)할 수 있습니다:
#
# .. code-block:: python
#
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             net = nn.DataParallel(net)
#     net.to(device)
#
# ``device`` 변수를 사용하여 사용가능한 GPU가 없을때도 훈련(training)이 잘 수행되도록 합니다.
# PyTorch는 아래와 같이 GPU 메모리에 데이터를 명시적으로 전송하도록 요구합니다:
#
# .. code-block:: python
#
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#
# 이제 이 코드는 CPU, 단일CPU 그리고 다중 GPU에 대한 훈련을 지원합니다. 특히, 
# Ray는 `분할(fractional) GPUs <https://docs.ray.io/en/master/using-ray-with-gpus.html#fractional-gpus>`_
# 또한 지원하므로 모델이 여전히 GPU 메모리에 적합(fit)하다면 여러 시도(trials)간 GPU를 공유할 수 있습니다.
# 이는 나중에 다시 소개하겠습니다.
#
# Ray Tune과 소통(communicating)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 가장 흥미로운 부분은 Ray Tune과 소통(communication) 하는 것입니다:
#
# .. code-block:: python
#
#     with tune.checkpoint_dir(epoch) as checkpoint_dir:
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         torch.save((net.state_dict(), optimizer.state_dict()), path)
#
#     tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
#
# 첫번째로 체크포인트를 저장하고 몇가지 지표(metrics)를 Ray Tune에 보고(report)합니다. 구체적으로,
# 검증데이터에 대한 손실(the validation loss)과 정확도(accuracy)를 Ray Tune에 전달합니다.
# Ray Tune은 이 지표(metrics)들을 사용해 어떤 하이퍼파라미터 구성이 최고의 결과를 도출하는지 결정할 수 있습니다.
# 이 지표(metrics)들을 사용해 나쁜 성능을 보이는 시도(trials)를 빨리 멈추어 자원(resources) 낭비를 막을수도 있습니다.
#
# 체크포인트 저장은 선택사항이지만
# `모수 기반 훈련(Population Based Training) <https://docs.ray.io/en/master/tune/tutorials/tune-advanced-tutorial.html>`_과
# 같은 고급(advanced) 스케쥴러(schedulers)를 사용한다면 필요한 기능입니다.
# 또한 체크포인트를 저장함으로써 나중에 훈련된 모델을 불러오고 테스트셋에서 검증(validate)할 수 있습니다.
#
# 훈련 함수 전문
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 전체 코드는 아래와 같습니다:


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

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
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
# 보시다시피 대부분의 코드는 원래의 예시에서 직접 조정되었습니다.
#
# 테스트셋 정확도(accuracy)
# -----------------
# 일반적으로 기계 학습 모델의 성능은 훈련에 사용되지 않고 보류된(hold-out) 테스트셋을 통해 테스트됩니다.
# 이 과정 또한 함수로 포장합니다:


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
# 또한 함수는 device 파라미터를 요구하므로 테스트셋 검증(validation)을 GPU에서 진행할 수 있습니다.
#
# 탐색범위(search space) 구성
# ----------------------------
# 마지막으로 아래의 예시와 같이 Ray Tune의 탐색범위(search space)를 정의해야합니다:
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
# ``tune.sample_from()`` 함수로 하이퍼파라미터를 얻기 위한  당신의 표본(sample) 방법(methods)를 정의할 수 있게 해줍니다.
# 위의 예시에서  ``l1`` 과 ``l2``파라미터는 4~256 범위의 2의 거듭제곱이어야 하므로 4, 8, 16, 32, 64, 128, 256 중 하나입니다.
# ``lr`` 즉 학습률(learning rate)은 0.0001~0.1 범위에서 균일확률로(uniformly) 추출(sampled)됩니다. 마지막으로,
# 배치 크기(batch size)는 2, 4, 8, 16 중 하나가 됩니다.
#
# 각각의 시도(trial)에서 RayTune은 각 파라미터들을 탐색범위(search spalces)내에서 무작위(randomly) 추출(sample)해 조합할 것입니다.
# 그런 다음 여러 모델을 병렬(parallel)로 훈련하고 이들중 가장 좋은 성능을 보이는 모델을 찾을것입니다.
# 또한 ``ASHAScheduler`` 를 사용해 나쁜 성능을 보이는 시도(trials)를 일찍 제거할것입니다.
#
# ``data_dir`` 파라미터를 설정해주기 위해 ``train_cifar`` 함수를 ``functools.partial`` 로 포장(wrap)합니다.
# 또한 Ray Tune에게 각각의 시도(trial)에서 어떤 자원(resources)이 사용가능한지 전달할 수 있습니다:
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
# 사용가능한 CPU의 수를 지정할 수 있습니다. 예시로 PyTorch ``DataLoader`` 인스턴스(instances) 의 ``num_workers`` 를 증가시키는것을 들수 있습니다.
# 선택한 GPU의 수는 각각의 시도(trial)에서 PyTorch에 표시됩니다.
# 각 시도(trials)는 요청되지 않은 GPU에 접근(access)할 수 없습니다 - 
# 따라서 동일한 자원(resource)을 사용하는 두 가지 시도(trial)에 대해 신경 쓸 필요가 없습니다.
#
# 또한 GPU 분할 지정도 가능하므로 ``gpus_per_trial=0.5`` 와 같은 것들도 사용할 수 있습니다.
# 그 후 각 시도(trials)들은 GPU를 서로 공유할 것입니다.
# 모델이 GPU 메모리에 여전히 적합(fit)한지 확인해야합니다.
#
# 모델을 훈련시킨뒤 가장 좋은성능을 보이는 모델을 찾고 체크포인트 파일로 훈련된 네트워크를 로드할것입니다.
# 그 후 테스트셋 정확도를 구하고 출력(printing)을 통해 모두 보고(report)합니다.
#
# main 함수의 전문은 아래와 같습니다:


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
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)


######################################################################
# 코드 실행시 출력은 아래와 같습니다:
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
# 대부분의 시도(trials)는 자원(resources)의 낭비를 막기위해 조기 종료(stopped early)됩니다.
# 가장 좋은성능을 보이는 시도는 검증데이터에 대한 정확도(validation accuracy)가 약 58%임을 테스트셋에서 확인할 수 있습니다.
#
# 끝났습니다! 이제 당신은 PyTorch 모델의 파라미터를 조정(tune)할 수 있습니다.
