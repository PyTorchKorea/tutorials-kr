# -*- coding: utf-8 -*-
"""
지식 증류 튜토리얼
===============================
저자: Alexandros Chariton (<https://github.com/AlexandrosChrtn>)
"""

######################################################################
# 지식 증류는 크고 계산 비용이 큰 모델로부터 지식을 작은 모델로 이전하는 기법입니다.
# 유효성을 크게 잃지 않으면서 작은 모델로 지식을 옮길 수 있어 연산 자원이 제한된 환경에 배포할 수 있습니다.
# 이로 인해 평가가 더 빠르고 효율적으로 이루어집니다.
#
# 이 튜토리얼에서는 경량 신경망의 정확도를 향상시키기 위한 여러 실험을 진행합니다.
# 더 강력한 네트워크를 교사(teacher)로 활용해 경량 학생(student) 네트워크를 개선하는 방식입니다.
# 경량 네트워크의 계산 비용과 속도는 변경되지 않으며, 우리는 순전히 가중치 학습 과정에만 개입합니다.
# 이 기법은 드론이나 모바일 기기처럼 연산 자원이 제한적인 장치에 유용합니다.
# 이 튜토리얼에서는 외부 패키지를 별도로 사용하지 않습니다. 필요한 기능은 `torch`와
# `torchvision`에서 제공합니다.
#
# 이 튜토리얼을 통해 배우게 될 내용:
#
# - 모델 클래스를 수정해 은닉 표현(hidden representations)을 추출하고 이를 추가 계산에 활용하는 방법
# - PyTorch의 학습 루프를 수정해 교차엔트로피 등 기존 손실에 추가 손실을 포함시키는 방법
# - 더 복잡한 모델을 교사로 사용해 경량 모델의 성능을 향상시키는 방법
#
# 사전 요구 사항
# ~~~~~~~~~~~~~
#
# * 1개의 GPU (메모리 4GB 권장)
# * PyTorch v2.0 이상
# * CIFAR-10 데이터셋 (스크립트가 다운로드하여 `./data`에 저장)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 현재 사용 가능한 `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`를 확인하고,
# 없으면 CPU를 사용하도록 설정합니다.
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

######################################################################
# CIFAR-10 불러오기
# ----------------
# CIFAR-10은 10개 클래스를 가진 널리 사용되는 이미지 데이터셋입니다. 목표는 각 입력 이미지에 대해 해당 클래스 중 하나를 예측하는 것입니다.
#
# (이미지 예시: /../_static/img/cifar10.png, 중앙 정렬)
#
# 입력 이미지는 RGB로 채널 수는 3이며 크기는 32×32입니다. 각 이미지는 0–255 범위의 3 × 32 × 32 = 3072개의 값으로 표현됩니다.
# 신경망에서는 활성화 함수 포화 방지와 수치 안정성 향상을 위해 입력을 정규화하는 것이 일반적입니다.
# 여기서는 채널별로 평균을 빼고 표준편차로 나누는 방식으로 정규화를 수행합니다.
# `mean=[0.485, 0.456, 0.406]` 및 `std=[0.229, 0.224, 0.225]` 값은 학습용으로 정한 CIFAR-10의 부분집합에서 계산된 평균/표준편차입니다.
# 테스트 세트에서도 이 값을 다시 계산하지 않고 동일하게 사용합니다. 학습 시 동일한 정규화로 훈련되었기 때문에 일관성을 유지하기 위함입니다.
# 현실적으로 테스트 데이터의 평균/표준편차는 미리 계산할 수 없는 경우가 많으므로 이러한 관행이 중요합니다.
# 검증(validation) 세트로 모델을 최적화한 후 별도의 테스트 세트로 최종 성능을 평가하여 하나의 지표에 편향된 모델 선택을 피합니다.

# 아래에서는 CIFAR-10 데이터를 전처리합니다. 배치 크기는 임의로 128을 사용합니다.
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

########################################################################
# 참고: 이 섹션은 빠른 결과를 원하는 CPU 사용자용입니다. 소규모 실험에만 사용하세요. GPU가 있다면 전체 코드가 훨씬 빠르게 동작합니다.
# 학습/테스트 데이터에서 처음 `num_images_to_keep`개 이미지만 선택하는 예시입니다.
#
#    .. code-block:: python
#
#       #from torch.utils.data import Subset
#       #num_images_to_keep = 2000
#       #train_dataset = Subset(train_dataset, range(min(num_images_to_keep, 50_000)))
#       #test_dataset = Subset(test_dataset, range(min(num_images_to_keep, 10_000)))

# 데이터로더(Dataloaders)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

######################################################################
# 모델 클래스 및 유틸리티 함수 정의
# --------------------------------------------
# 다음으로 모델 클래스를 정의합니다. 몇몇 사용자 설정 파라미터를 지정해야 하며, 공정한 비교를 위해 실험 전반에서 필터 수는 고정합니다.
# 두 아키텍처 모두 합성곱 계층을 특징 추출기로 사용하고 그 뒤에 10개 클래스를 분류하는 분류기가 붙는 CNN입니다.
# 학생 모델 쪽이 필터와 뉴런 수가 더 적습니다.

# 교사(teacher)로 사용할 더 깊은 신경망 클래스:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 학생(student)으로 사용할 경량 신경망 클래스:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

######################################################################
# 원래 분류 과제 결과를 생성하고 평가하기 위해 두 가지 함수를 사용합니다.
# `train` 함수는 다음 인자를 받습니다:
#
# - `model`: 학습(가중치 업데이트)할 모델 인스턴스
# - `train_loader`: 모델에 데이터를 공급하는 데이터로더
# - `epochs`: 데이터셋을 반복할 횟수(에포크 수)
# - `learning_rate`: 업데이트 크기를 결정하는 학습률
# - `device`: 연산을 수행할 장치(CPU 또는 GPU)
#
# 테스트 함수는 유사하며 `test_loader`를 사용해 테스트 세트의 이미지를 불러옵니다.
#
# (이미지: /../_static/img/knowledge_distillation/ce_only.png, 중앙 정렬)
#
# 두 네트워크를 교차 엔트로피로 학습합니다. 학생 모델을 기준선(baseline)으로 사용합니다.

def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 입력: batch_size만큼의 이미지 묶음
            # 라벨: 각 이미지의 클래스를 정수로 나타내는 길이 batch_size의 벡터
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 출력: batch_size x num_classes 크기의 텐서
            # 라벨: 실제 이미지 라벨 (길이 batch_size의 벡터)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

######################################################################
# 교차 엔트로피 실험 실행
# ------------------
# 재현성을 위해 `torch.manual_seed`를 설정합니다. 서로 다른 방법으로 학습한 네트워크를 공정하게 비교하려면
# 동일한 가중치로 초기화하는 것이 합리적입니다.
# 먼저 교사 네트워크를 교차 엔트로피로 학습합니다:

torch.manual_seed(42)
nn_deep = DeepNN(num_classes=10).to(device)
train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_deep = test(nn_deep, test_loader, device)

# 경량 네트워크 인스턴스 생성:
torch.manual_seed(42)
nn_light = LightNN(num_classes=10).to(device)

######################################################################
# 성능 비교를 위해 경량 네트워크 모델을 하나 더 생성합니다.
# 역전파는 가중치 초기화에 민감하므로 두 네트워크가 동일한 초기화인지 확인해야 합니다.

torch.manual_seed(42)
new_nn_light = LightNN(num_classes=10).to(device)

######################################################################
# 첫 네트워크의 복사본을 만들었는지 확인하기 위해 첫 레이어의 노름(norm)을 검사합니다.
# 값이 일치하면 두 네트워크가 동일하다고 판단할 수 있습니다.

# 초기 경량 모델의 첫 레이어 노름 출력
print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
# 새 경량 모델의 첫 레이어 노름 출력
print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())

######################################################################
# 각 모델의 전체 파라미터 수를 출력합니다:
total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
print(f"DeepNN parameters: {total_params_deep}")
total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
print(f"LightNN parameters: {total_params_light}")

######################################################################
# 교차 엔트로피 손실로 경량 네트워크를 학습하고 테스트합니다:
train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_light_ce = test(nn_light, test_loader, device)

######################################################################
# 테스트 정확도를 기준으로 교사 모델과 학생 모델을 비교할 수 있습니다. 지금까지 학생의 성능은 교사의 개입 없이 학생 자체의 성능입니다.
# 아래 출력으로 지금까지의 지표를 확인할 수 있습니다:

print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")

######################################################################
# 지식 증류 실행
# --------------------------
# 이제 교사를 활용해 학생 네트워크의 테스트 정확도를 향상시켜 보겠습니다.
# 지식 증류는 두 네트워크가 동일한 클래스에 대한 확률 분포를 출력한다는 점에 기반한 간단한 기법입니다.
# 따라서 두 네트워크는 동일한 수의 출력 뉴런을 가집니다.
# 이 방법은 교차 엔트로피 손실에 교사의 softmax 출력(soft targets)에 기반한 추가 손실을 결합합니다.
# 교사 출력의 활성화 값에는 학생이 학습 과정에서 활용할 수 있는 추가 정보가 담겨 있다고 가정합니다.
# 원 논문은 soft target의 작은 확률들 간 비율을 활용하면 유사성 구조를 잘 학습하는 데 도움이 된다고 제안합니다.
# 예를 들어 CIFAR-10에서 트럭은 자동차나 비행기로 오인될 수 있지만 개로 오인될 가능성은 낮습니다.
# 따라서 전체 출력 분포에 유용한 정보가 포함되어 있다고 보는 것이 합리적입니다.
# 그러나 교차 엔트로피만으로는 비예측 클래스의 활성화가 너무 작아 그래디언트가 충분히 전달되지 못하는 문제가 있습니다.
#
# 교사-학생 동작을 도입하는 헬퍼 함수를 정의할 때는 몇 가지 추가 파라미터가 필요합니다:
#
# - `T`(온도): 출력 분포의 평활도를 조절합니다. `T`가 크면 작은 확률 값들이 상대적으로 더 커집니다.
# - `soft_target_loss_weight`: soft target 손실에 부여할 가중치
# - `ce_loss_weight`: 교차 엔트로피 손실에 부여할 가중치
#
# (이미지: /../_static/img/knowledge_distillation/distillation_output_loss.png, 중앙 정렬)
#
# 증류 손실은 네트워크의 로짓에서 계산되며 그래디언트는 학생에게만 전달됩니다.

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # 교사 모델을 평가 모드로 설정
    student.train() # 학생 모델을 학습 모드로 설정

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 교사 모델로 순전파 수행 — 교사 가중치는 변경하지 않으므로 그래디언트를 저장하지 않습니다
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # 학생 모델로 순전파 수행
            student_logits = student(inputs)

            # 학생의 로짓을 부드럽게 만들기 위해 softmax를 적용한 뒤 log()를 취합니다
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # soft target 손실을 계산합니다. 논문 권고대로 T**2로 스케일링합니다
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # 실제 라벨에 대한 손실을 계산합니다
            label_loss = ce_loss(student_logits, labels)

            # 두 손실의 가중 합을 계산합니다
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# `train_knowledge_distillation`를 온도 2로 적용합니다. 임의로 CE 가중치를 0.75, 증류 손실 가중치를 0.25로 설정했습니다.
train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

# 증류 후 교사를 사용한 경우와 사용하지 않은 경우의 학생 테스트 정확도를 비교합니다
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")

######################################################################
# 코사인 손실 최소화 실험 실행
# ----------------------------
# softmax 평활도와 손실 계수를 조절하는 온도 파라미터를 자유롭게 실험해 보세요.
# 신경망에서는 추가 손실 함수를 주 목적에 더해 포함시켜 일반화 성능을 개선할 수 있습니다.
# 이번에는 출력층이 아니라 은닉 상태(hidden states)에 초점을 맞춰 학생 모델에 추가 목적을 도입해 보겠습니다.
# 단순한 손실을 최소화하면 분류기에 전달되는 평탄화된 벡터들이 손실이 줄어들수록 서로 더 유사해지도록 유도됩니다.
# 교사는 가중치를 업데이트하지 않으므로 최소화 과정은 학생의 가중치에만 영향을 줍니다.
# 이 방법의 근거는 교사 모델이 더 우수한 내부 표현을 가지고 있다는 가정이며, 학생이 외부 개입 없이는 얻기 어려운 표현을 모방하도록 합니다.
# 다만 네트워크 구조와 학습 능력 차이로 인해 이것이 항상 도움이 된다고 보장할 수는 없습니다.
# 성분 순서만 다른 표현(permutation)도 동일하게 효율적일 수 있습니다.
# 간단한 실험을 통해 영향을 확인해 보겠습니다.
# CosineEmbeddingLoss를 사용합니다 (수식 참조).
#
# 출력층에 증류를 적용할 때는 출력 뉴런 수가 같지만, 합성곱 뒤의 은닉층에서는 크기가 다를 수 있습니다.
# 평탄화한 후 차원이 달라지므로 손실 함수 입력 차원을 맞춰줘야 합니다. 이를 위해 교사 쪽에 평균 풀링을 적용해 차원을 줄입니다.
# 모델 클래스를 수정하거나 새 클래스를 만들어 forward가 로짓과 평탄화된 은닉 표현을 모두 반환하도록 합니다.

class ModifiedDeepNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, 2)
        return x, flattened_conv_output_after_pooling

# 학생 클래스도 튜플을 반환하도록 만들되 학생 쪽에서는 평탄화 후 풀링을 적용하지 않습니다.
class ModifiedLightNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        return x, flattened_conv_output

# 수정된 심층 네트워크를 처음부터 학습할 필요는 없습니다. 이미 학습된 인스턴스에서 가중치를 불러옵니다.
modified_nn_deep = ModifiedDeepNNCosine(num_classes=10).to(device)
modified_nn_deep.load_state_dict(nn_deep.state_dict())

# 다시 한 번 두 네트워크의 첫 레이어 노름이 동일한지 확인하세요.
print("Norm of 1st layer for deep_nn:", torch.norm(nn_deep.features[0].weight).item())
print("Norm of 1st layer for modified_deep_nn:", torch.norm(modified_nn_deep.features[0].weight).item())

# 다른 경량 인스턴스와 동일한 시드로 수정된 경량 네트워크를 초기화합니다. 코사인 손실 최소화의 효과를 확인하기 위해 처음부터 학습합니다.
torch.manual_seed(42)
modified_nn_light = ModifiedLightNNCosine(num_classes=10).to(device)
print("Norm of 1st layer:", torch.norm(modified_nn_light.features[0].weight).item())

######################################################################
# 모델이 `(logits, hidden_representation)` 튜플을 반환하므로 학습 루프를 변경해야 합니다. 샘플 입력 텐서로 텐서들의 형상을 확인할 수 있습니다.

# 샘플 입력 텐서를 생성합니다.
sample_input = torch.randn(128, 3, 32, 32).to(device) # Batch size: 128, Filters: 3, Image size: 32x32

# 학생 모델에 입력을 통과시켜 텐서들의 형상을 확인합니다.
logits, hidden_representation = modified_nn_light(sample_input)

print("Student logits shape:", logits.shape) # batch_size x total_classes
print("Student hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

# 교사 모델에도 입력을 통과시켜 형상을 확인합니다.
logits, hidden_representation = modified_nn_deep(sample_input)

print("Teacher logits shape:", logits.shape) # batch_size x total_classes
print("Teacher hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

######################################################################
# 이 예제에서 `hidden_representation_size`는 `1024`입니다. 학생의 최종 합성곱 층을 평탄화한 특징 맵이며 분류기의 입력으로 사용됩니다.
# 교사도 `avg_pool1d`로 차원을 맞춰 `1024`로 설정했습니다.
# 여기서 적용되는 손실은 학생의 분류기에는 영향을 주지 않고 분류기 이전의 가중치에만 영향을 미칩니다.
# 수정된 학습 루프는 다음과 같습니다.
# (이미지: /../_static/img/knowledge_distillation/cosine_loss_distillation.png)
# 코사인 손실 최소화에서는 두 표현의 코사인 유사도를 최대화하도록 학생에게 그래디언트를 되돌려 주는 것이 목표입니다.

def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # 교사 모델을 평가 모드로 설정
    student.train() # 학생 모델을 학습 모드로 설정

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 교사 모델로 순전파를 수행하고 은닉 표현만 보관합니다
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(inputs)

            # 학생 모델로 순전파 수행
            student_logits, student_hidden_representation = student(inputs)

            # 코사인 손실을 계산합니다. 타깃은 모두 1로 이루어진 벡터이며, 손실을 최소화하면 코사인 유사도가 증가합니다.
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))

            # 실제 라벨에 대한 손실을 계산합니다
            label_loss = ce_loss(student_logits, labels)

            # 두 손실의 가중 합을 계산합니다
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

######################################################################
# 같은 이유로 테스트 함수도 수정해야 합니다. 여기서는 모델이 반환하는 은닉 표현은 무시합니다.

def test_multiple_outputs(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs) # 튜플의 두 번째 텐서는 무시합니다
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

######################################################################
# 지식 증류와 코사인 손실 최소화를 동일한 함수에 결합하는 것도 가능합니다. 교사-학생 패러다임에서는 방법을 결합해 성능을 높이는 경우가 흔합니다.
# 우선 간단한 학습-테스트 세션을 실행해봅니다.

# 교차 엔트로피 + 코사인 손실 학습 실행
train_cosine_loss(teacher=modified_nn_deep, student=modified_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_cosine_loss = test_multiple_outputs(modified_nn_light, test_loader, device)

######################################################################
# 중간 regressor 실험 실행
# --------------------------
# 단순한 최소화가 항상 좋은 결과를 보장하지 않는 이유는 여러 가지가 있으며, 그중 하나는 벡터 차원 문제입니다.
# 차원이 높은 벡터에서는 코사인 유사도가 유클리드 거리보다 더 잘 동작하는 경향이 있지만,
# 여기서는 각 벡터가 1024차원이므로 의미 있는 유사성을 추출하기가 어렵습니다.
# 또한 교사와 학생의 은닉 표현을 1:1로 맞추도록 강제하는 것은 이론적으로 정당화되기 어렵습니다.
# 마지막 예제로 regressor라는 추가 네트워크를 포함해 학습 개입 방법을 보여드리겠습니다.
# 목표는 교사와 학생의 합성곱 층 이후 특징 맵을 추출한 뒤 이를 맞추는 것입니다.
# 이를 위해 네트워크 사이에 학습 가능한 regressor를 두어 매칭 과정을 돕습니다.
# regressor는 특징 맵의 차원을 맞춰 손실 함수를 정의할 수 있게 합니다.
# 이러한 손실 함수는 학생의 가중치를 변경하는 그래디언트를 전파할 수 있는 학습 경로를 제공합니다.
# 합성곱 특징 추출기만 통과시켜 출력의 형상을 확인합니다.

convolutional_fe_output_student = nn_light.features(sample_input)
convolutional_fe_output_teacher = nn_deep.features(sample_input)

print("Student's feature extractor output shape: ", convolutional_fe_output_student.shape)
print("Teacher's feature extractor output shape: ", convolutional_fe_output_teacher.shape)

######################################################################
# 교사 모델은 32개의 필터, 학생 모델은 16개의 필터를 가집니다.
# 학생의 특징 맵을 교사의 특징 맵 형태로 변환하는 학습 가능한 레이어를 포함합니다.
# 실제로는 경량 클래스에서 중간 regressor 뒤의 은닉 상태를 반환하도록 수정하고,
# 교사 클래스는 풀링이나 평탄화 없이 최종 합성곱 층의 출력을 반환하도록 합니다.
# (이미지: /../_static/img/knowledge_distillation/fitnets_knowledge_distill.png)
# 학습 가능한 레이어가 중간 텐서들의 형상을 맞추면 MSE를 적절히 정의할 수 있습니다.

class ModifiedDeepNNRegressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        conv_feature_map = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, conv_feature_map

class ModifiedLightNNRegressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 추가 regressor(여기서는 합성곱 레이어)를 포함합니다
        self.regressor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        regressor_output = self.regressor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, regressor_output

######################################################################
# 그 후 학습 루프를 다시 업데이트합니다. 이번에는 학생의 regressor 출력과 교사의 특징 맵을 추출하고,
# 이들 텐서에 대해 MSE를 계산한 뒤(형상이 동일하므로 적절히 정의됨) 해당 손실에 대해 그래디언트를 역전파합니다.
# 이는 분류 과제의 교차 엔트로피 손실과 함께 적용됩니다.

def train_mse_loss(teacher, student, train_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # 교사 모델을 평가 모드로 설정
    student.train() # 학생 모델을 학습 모드로 설정

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 다시 교사의 로짓은 무시합니다
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)

            # 학생 모델로 순전파 수행
            student_logits, regressor_feature_map = student(inputs)

            # 손실을 계산합니다
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)

            # 실제 라벨에 대한 손실
            label_loss = ce_loss(student_logits, labels)

            # 두 손실의 가중 합
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# 테스트 함수는 이전과 동일합니다. 정확도 측정에서는 실제 출력만 중요합니다.

# ModifiedLightNNRegressor(수정된 경량 회귀기)를 초기화합니다
torch.manual_seed(42)
modified_nn_light_reg = ModifiedLightNNRegressor(num_classes=10).to(device)

# 수정된 심층 네트워크는 처음부터 학습할 필요 없이 학습된 인스턴스에서 가중치를 불러옵니다
modified_nn_deep_reg = ModifiedDeepNNRegressor(num_classes=10).to(device)
modified_nn_deep_reg.load_state_dict(nn_deep.state_dict())

# 다시 학습하고 테스트합니다
train_mse_loss(teacher=modified_nn_deep_reg, student=modified_nn_light_reg, train_loader=train_loader, epochs=10, learning_rate=0.001, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_mse_loss = test_multiple_outputs(modified_nn_light_reg, test_loader, device)

######################################################################
# 최종 방법은 `CosineLoss`보다 더 잘 동작할 것으로 기대됩니다. 그 이유는 교사와 학생 사이에 학습 가능한 레이어를 두어
# 학생이 교사의 표현을 단순히 복제하도록 강요받지 않고 학습에 여유를 가질 수 있도록 했기 때문입니다.
# 추가 네트워크 도입은 힌트 기반 증류(hint-based distillation)의 핵심 아이디어입니다.

print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
print(f"Student accuracy with CE + CosineLoss: {test_accuracy_light_ce_and_cosine_loss:.2f}%")
print(f"Student accuracy with CE + RegressorMSE: {test_accuracy_light_ce_and_mse_loss:.2f}%")

######################################################################
# 결론
# --------------------------------------------
# 위 방법들 중 어느 것도 모델의 파라미터 수나 추론 시간(inference time)을 증가시키지 않습니다.
# 따라서 성능 향상은 학습 시 그래디언트 계산이라는 작은 비용으로 얻을 수 있습니다.
# 머신러닝 응용에서는 보통 학습은 배포 전에 이루어지므로 추론 시간이 더 중요합니다.
# 만약 경량 모델이 여전히 배포하기에 무겁다면, 사후 양자화(post-training quantization)와 같은 기법을 적용할 수 있습니다.
# 추가 손실은 분류뿐 아니라 다양한 과제에 적용할 수 있으며, 계수(coefficients), 온도(temperature), 뉴런 수 등을 실험해 보세요.
# 다만 뉴런/필터 수를 변경하면 형태(shape) 불일치가 발생할 수 있음을 유의하세요.
#
# 추가 정보:
# Hinton, G., Vinyals, O., Dean, J.: "Distilling the knowledge in a neural network" (NIPS Deep Learning Workshop, 2015). https://arxiv.org/abs/1503.02531
# Romero, A., Ballas, N., Kahou, S.E., Chassang, A., Gatta, C., Bengio, Y.: "Fitnets: Hints for thin deep nets" (ICLR, 2015). https://arxiv.org/abs/1412.6550
# -*- coding: utf-8 -*-
"""
Knowledge Distillation Tutorial
===============================
**Author**: `Alexandros Chariton <https://github.com/AlexandrosChrtn>`_
""" 

######################################################################
# Knowledge distillation is a technique that enables knowledge transfer from large, computationally expensive
# models to smaller ones without losing validity. This allows for deployment on less powerful
# hardware, making evaluation faster and more efficient. 
#
# In this tutorial, we will run a number of experiments focused at improving the accuracy of a
# lightweight neural network, using a more powerful network as a teacher.
# The computational cost and the speed of the lightweight network will remain unaffected,
# our intervention only focuses on its weights, not on its forward pass.
# Applications of this technology can be found in devices such as drones or mobile phones.
# In this tutorial, we do not use any external packages as everything we need is available in ``torch`` and
# ``torchvision``.
#
# In this tutorial, you will learn:
#
# - How to modify model classes to extract hidden representations and use them for further calculations
# - How to modify regular train loops in PyTorch to include additional losses on top of, for example, cross-entropy for classification 
# - How to improve the performance of lightweight models by using more complex models as teachers
#
# Prerequisites
# ~~~~~~~~~~~~~
#
# * 1 GPU, 4GB of memory
# * PyTorch v2.0 or later 
# * CIFAR-10 dataset (downloaded by the script and saved in a directory called ``/data``)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Check if the current `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# is available, and if not, use the CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

######################################################################
# Loading CIFAR-10
# ----------------
# CIFAR-10 is a popular image dataset with ten classes. Our objective is to predict one of the following classes for each input image.
#
# .. figure:: /../_static/img/cifar10.png 
#    :align: center
#    
#    Example of CIFAR-10 images
#
# The input images are RGB, so they have 3 channels and are 32x32 pixels. Basically, each image is described by 3 x 32 x 32 = 3072 numbers ranging from 0 to 255.
# A common practice in neural networks is to normalize the input, which is done for multiple reasons,
# including avoiding saturation in commonly used activation functions and increasing numerical stability.
# Our normalization process consists of subtracting the mean and dividing by the standard deviation along each channel.
# The tensors "mean=[0.485, 0.456, 0.406]" and "std=[0.229, 0.224, 0.225]" were already computed,
# and they represent the mean and standard deviation of each channel in the
# predefined subset of CIFAR-10 intended to be the training set.
# Notice how we use these values for the test set as well, without recomputing the mean and standard deviation from scratch.
# This is because the network was trained on features produced by subtracting and dividing the numbers above, and we want to maintain consistency.
# Furthermore, in real life, we would not be able to compute the mean and standard deviation of the test set since,
# under our assumptions, this data would not be accessible at that point.
# 
# As a closing point, we often refer to this held-out set as the validation set, and we use a separate set,
# called the test set, after optimizing a model's performance on the validation set.
# This is done to avoid selecting a model based on the greedy and biased optimization of a single metric.

# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

########################################################################
# .. note:: This section is for CPU users only who are interested in quick results. Use this option only if you're interested in a small scale experiment. Keep in mind the code should run fairly quickly using any GPU. Select only the first ``num_images_to_keep`` images from the train/test dataset
#
#    .. code-block:: python
#
#       #from torch.utils.data import Subset
#       #num_images_to_keep = 2000
#       #train_dataset = Subset(train_dataset, range(min(num_images_to_keep, 50_000)))
#       #test_dataset = Subset(test_dataset, range(min(num_images_to_keep, 10_000)))

#Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

######################################################################
# Defining model classes and utility functions
# --------------------------------------------
# Next, we need to define our model classes. Several user-defined parameters need to be set here. We use two different architectures, keeping the number of filters fixed across our experiments to ensure fair comparisons.
# Both architectures are Convolutional Neural Networks (CNNs) with a different number of convolutional layers that serve as feature extractors, followed by a classifier with 10 classes. 
# The number of filters and neurons is smaller for the students.

# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

######################################################################
# We employ 2 functions to help us produce and evaluate the results on our original classification task.
# One function is called ``train`` and takes the following arguments:
#
# - ``model``: A model instance to train (update its weights) via this function.
# - ``train_loader``: We defined our ``train_loader`` above, and its job is to feed the data into the model.
# - ``epochs``: How many times we loop over the dataset.
# - ``learning_rate``: The learning rate determines how large our steps towards convergence should be. Too large or too small steps can be detrimental.
# - ``device``: Determines the device to run the workload on. Can be either CPU or GPU depending on availability.
#
# Our test function is similar, but it will be invoked with ``test_loader`` to load images from the test set.
#
# .. figure:: /../_static/img/knowledge_distillation/ce_only.png 
#    :align: center
#    
#    Train both networks with Cross-Entropy. The student will be used as a baseline:
#

def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

######################################################################
# Cross-entropy runs
# ------------------
# For reproducibility, we need to set the torch manual seed. We train networks using different methods, so to compare them fairly,
# it makes sense to initialize the networks with the same weights.
# Start by training the teacher network using cross-entropy:

torch.manual_seed(42)
nn_deep = DeepNN(num_classes=10).to(device)
train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_deep = test(nn_deep, test_loader, device)

# Instantiate the lightweight network:
torch.manual_seed(42)
nn_light = LightNN(num_classes=10).to(device)

######################################################################
# We instantiate one more lightweight network model to compare their performances.
# Back propagation is sensitive to weight initialization,
# so we need to make sure these two networks have the exact same initialization.

torch.manual_seed(42)
new_nn_light = LightNN(num_classes=10).to(device)

######################################################################
# To ensure we have created a copy of the first network, we inspect the norm of its first layer.
# If it matches, then we are safe to conclude that the networks are indeed the same.

# Print the norm of the first layer of the initial lightweight model
print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
# Print the norm of the first layer of the new lightweight model
print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())

######################################################################
# Print the total number of parameters in each model:
total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
print(f"DeepNN parameters: {total_params_deep}")
total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
print(f"LightNN parameters: {total_params_light}")

######################################################################
# Train and test the lightweight network with cross entropy loss:
train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_light_ce = test(nn_light, test_loader, device)

######################################################################
# As we can see, based on test accuracy, we can now compare the deeper network that is to be used as a teacher with the lightweight network that is our supposed student. So far, our student has not intervened with the teacher, therefore this performance is achieved by the student itself.
# The metrics so far can be seen with the following lines:

print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")

######################################################################
# Knowledge distillation run
# --------------------------
# Now let's try to improve the test accuracy of the student network by incorporating the teacher.
# Knowledge distillation is a straightforward technique to achieve this,
# based on the fact that both networks output a probability distribution over our classes.
# Therefore, the two networks share the same number of output neurons.
# The method works by incorporating an additional loss into the traditional cross entropy loss,
# which is based on the softmax output of the teacher network.
# The assumption is that the output activations of a properly trained teacher network carry additional information that can be leveraged by a student network during training.
# The original work suggests that utilizing ratios of smaller probabilities in the soft targets can help achieve the underlying objective of deep neural networks,
# which is to create a similarity structure over the data where similar objects are mapped closer together.
# For example, in CIFAR-10, a truck could be mistaken for an automobile or airplane,
# if its wheels are present, but it is less likely to be mistaken for a dog. 
# Therefore, it makes sense to assume that valuable information resides not only in the top prediction of a properly trained model but in the entire output distribution.
# However, cross entropy alone does not sufficiently exploit this information as the activations for non-predicted classes
# tend to be so small that propagated gradients do not meaningfully change the weights to construct this desirable vector space.
#
# As we continue defining our first helper function that introduces a teacher-student dynamic, we need to include a few extra parameters:
# 
# - ``T``: Temperature controls the smoothness of the output distributions. Larger ``T`` leads to smoother distributions, thus smaller probabilities get a larger boost.
# - ``soft_target_loss_weight``: A weight assigned to the extra objective we're about to include.
# - ``ce_loss_weight``: A weight assigned to cross-entropy. Tuning these weights pushes the network towards optimizing for either objective.
#
# .. figure:: /../_static/img/knowledge_distillation/distillation_output_loss.png 
#    :align: center
#    
#    Distillation loss is calculated from the logits of the networks. It only returns gradients to the student:
#

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

# Compare the student test accuracy with and without the teacher, after distillation
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")

######################################################################
# Cosine loss minimization run
# ----------------------------
# Feel free to play around with the temperature parameter that controls the softness of the softmax function and the loss coefficients.
# In neural networks, it is easy to include additional loss functions to the main objectives to achieve goals like better generalization.
# Let's try including an objective for the student, but now let's focus on their hidden states rather than their output layers.
# Our goal is to convey information from the teacher's representation to the student by including a naive loss function,
# whose minimization implies that the flattened vectors that are subsequently passed to the classifiers have become more *similar* as the loss decreases.
# Of course, the teacher does not update its weights, so the minimization depends only on the student's weights.
# The rationale behind this method is that we are operating under the assumption that the teacher model has a better internal representation that is
# unlikely to be achieved by the student without external intervention, therefore we artificially push the student to mimic the internal representation of the teacher.
# Whether or not this will end up helping the student is not straightforward, though, because pushing the lightweight network
# to reach this point could be a good thing, assuming that we have found an internal representation that leads to better test accuracy,
# but it could also be harmful because the networks have different architectures and the student does not have the same learning capacity as the teacher.
# In other words, there is no reason for these two vectors, the student's and the teacher's to match per component.
# The student could reach an internal representation that is a permutation of the teacher's and it would be just as efficient.
# Nonetheless, we can still run a quick experiment to figure out the impact of this method.
# We will be using the ``CosineEmbeddingLoss`` which is given by the following formula:
#
# .. figure:: /../_static/img/knowledge_distillation/cosine_embedding_loss.png 
#    :align: center
#    :width: 450px
#    
#    Formula for CosineEmbeddingLoss
#
# Obviously, there is one thing that we need to resolve first.
# When we applied distillation to the output layer we mentioned that both networks have the same number of neurons, equal to the number of classes.
# However, this is not the case for the layer following our convolutional layers. Here, the teacher has more neurons than the student
# after the flattening of the final convolutional layer. Our loss function accepts two vectors of equal dimensionality as inputs,
# therefore we need to somehow match them. We will solve this by including an average pooling layer after the teacher's convolutional layer to reduce its dimensionality to match that of the student.
#
# To proceed, we will modify our model classes, or create new ones.
# Now, the forward function returns not only the logits of the network but also the flattened hidden representation after the convolutional layer. We include the aforementioned pooling for the modified teacher.

class ModifiedDeepNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, 2)
        return x, flattened_conv_output_after_pooling

# Create a similar student class where we return a tuple. We do not apply pooling after flattening.
class ModifiedLightNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        return x, flattened_conv_output

# We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
modified_nn_deep = ModifiedDeepNNCosine(num_classes=10).to(device)
modified_nn_deep.load_state_dict(nn_deep.state_dict())

# Once again ensure the norm of the first layer is the same for both networks
print("Norm of 1st layer for deep_nn:", torch.norm(nn_deep.features[0].weight).item())
print("Norm of 1st layer for modified_deep_nn:", torch.norm(modified_nn_deep.features[0].weight).item())

# Initialize a modified lightweight network with the same seed as our other lightweight instances. This will be trained from scratch to examine the effectiveness of cosine loss minimization.
torch.manual_seed(42)
modified_nn_light = ModifiedLightNNCosine(num_classes=10).to(device)
print("Norm of 1st layer:", torch.norm(modified_nn_light.features[0].weight).item())

######################################################################
# Naturally, we need to change the train loop because now the model returns a tuple ``(logits, hidden_representation)``. Using a sample input tensor
# we can print their shapes.

# Create a sample input tensor
sample_input = torch.randn(128, 3, 32, 32).to(device) # Batch size: 128, Filters: 3, Image size: 32x32

# Pass the input through the student
logits, hidden_representation = modified_nn_light(sample_input)

# Print the shapes of the tensors
print("Student logits shape:", logits.shape) # batch_size x total_classes
print("Student hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

# Pass the input through the teacher
logits, hidden_representation = modified_nn_deep(sample_input)

# Print the shapes of the tensors
print("Teacher logits shape:", logits.shape) # batch_size x total_classes
print("Teacher hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

######################################################################
# In our case, ``hidden_representation_size`` is ``1024``. This is the flattened feature map of the final convolutional layer of the student and as you can see,
# it is the input for its classifier. It is ``1024`` for the teacher too, because we made it so with ``avg_pool1d`` from ``2048``.
# The loss applied here only affects the weights of the student prior to the loss calculation. In other words, it does not affect the classifier of the student.
# The modified training loop is the following:
#
# .. figure:: /../_static/img/knowledge_distillation/cosine_loss_distillation.png 
#    :align: center
#    
#    In Cosine Loss minimization, we want to maximize the cosine similarity of the two representations by returning gradients to the student:
#

def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model and keep only the hidden representation
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(inputs)

            # Forward pass with the student model
            student_logits, student_hidden_representation = student(inputs)

            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

######################################################################
#We need to modify our test function for the same reason. Here we ignore the hidden representation returned by the model.

def test_multiple_outputs(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs) # Disregard the second tensor of the tuple
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

######################################################################
# In this case, we could easily include both knowledge distillation and cosine loss minimization in the same function. It is common to combine methods to achieve better performance in teacher-student paradigms.
# For now, we can run a simple train-test session.

# Train and test the lightweight network with cross entropy loss
train_cosine_loss(teacher=modified_nn_deep, student=modified_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_cosine_loss = test_multiple_outputs(modified_nn_light, test_loader, device)

######################################################################
# Intermediate regressor run
# --------------------------
# Our naive minimization does not guarantee better results for several reasons, one being the dimensionality of the vectors.
# Cosine similarity generally works better than Euclidean distance for vectors of higher dimensionality,
# but we were dealing with vectors with 1024 components each, so it is much harder to extract meaningful similarities.
# Furthermore, as we mentioned, pushing towards a match of the hidden representation of the teacher and the student is not supported by theory.
# There are no good reasons why we should be aiming for a 1:1 match of these vectors.
# We will provide a final example of training intervention by including an extra network called regressor.
# The objective is to first extract the feature map of the teacher after a convolutional layer,
# then extract a feature map of the student after a convolutional layer, and finally try to match these maps.
# However, this time, we will introduce a regressor between the networks to facilitate the matching process.
# The regressor will be trainable and ideally will do a better job than our naive cosine loss minimization scheme.
# Its main job is to match the dimensionality of these feature maps so that we can properly define a loss function between the teacher and the student.
# Defining such a loss function provides a teaching "path," which is basically a flow to back-propagate gradients that will change the student's weights.
# Focusing on the output of the convolutional layers right before each classifier for our original networks, we have the following shapes:
#

# Pass the sample input only from the convolutional feature extractor
convolutional_fe_output_student = nn_light.features(sample_input)
convolutional_fe_output_teacher = nn_deep.features(sample_input)

# Print their shapes
print("Student's feature extractor output shape: ", convolutional_fe_output_student.shape)
print("Teacher's feature extractor output shape: ", convolutional_fe_output_teacher.shape)

######################################################################
# We have 32 filters for the teacher and 16 filters for the student.
# We will include a trainable layer that converts the feature map of the student to the shape of the feature map of the teacher.
# In practice, we modify the lightweight class to return the hidden state after an intermediate regressor that matches the sizes of the convolutional
# feature maps and the teacher class to return the output of the final convolutional layer without pooling or flattening.
#
# .. figure:: /../_static/img/knowledge_distillation/fitnets_knowledge_distill.png 
#    :align: center
#    
#    The trainable layer matches the shapes of the intermediate tensors and Mean Squared Error (MSE) is properly defined:
#

class ModifiedDeepNNRegressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        conv_feature_map = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, conv_feature_map

class ModifiedLightNNRegressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Include an extra regressor (in our case linear)
        self.regressor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        regressor_output = self.regressor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, regressor_output

######################################################################
# After that, we have to update our train loop again. This time, we extract the regressor output of the student, the feature map of the teacher,
# we calculate the ``MSE`` on these tensors (they have the exact same shape so it's properly defined) and we back propagate gradients based on that loss,
# in addition to the regular cross entropy loss of the classification task.

def train_mse_loss(teacher, student, train_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Again ignore teacher logits
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)

            # Forward pass with the student model
            student_logits, regressor_feature_map = student(inputs)

            # Calculate the loss
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Notice how our test function remains the same here with the one we used in our previous case. We only care about the actual outputs because we measure accuracy.

# Initialize a ModifiedLightNNRegressor
torch.manual_seed(42)
modified_nn_light_reg = ModifiedLightNNRegressor(num_classes=10).to(device)

# We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
modified_nn_deep_reg = ModifiedDeepNNRegressor(num_classes=10).to(device)
modified_nn_deep_reg.load_state_dict(nn_deep.state_dict())

# Train and test once again
train_mse_loss(teacher=modified_nn_deep_reg, student=modified_nn_light_reg, train_loader=train_loader, epochs=10, learning_rate=0.001, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_mse_loss = test_multiple_outputs(modified_nn_light_reg, test_loader, device)

######################################################################
# It is expected that the final method will work better than ``CosineLoss`` because now we have allowed a trainable layer between the teacher and the student,
# which gives the student some wiggle room when it comes to learning, rather than pushing the student to copy the teacher's representation.
# Including the extra network is the idea behind hint-based distillation.

print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
print(f"Student accuracy with CE + CosineLoss: {test_accuracy_light_ce_and_cosine_loss:.2f}%")
print(f"Student accuracy with CE + RegressorMSE: {test_accuracy_light_ce_and_mse_loss:.2f}%")

######################################################################
# Conclusion
# --------------------------------------------
# None of the methods above increases the number of parameters for the network or inference time,
# so the performance increase comes at the little cost of calculating gradients during training.
# In ML applications, we mostly care about inference time because training happens before the model deployment.
# If our lightweight model is still too heavy for deployment, we can apply different ideas, such as post-training quantization.
# Additional losses can be applied in many tasks, not just classification, and you can experiment with quantities like coefficients,
# temperature, or number of neurons. Feel free to tune any numbers in the tutorial above,
# but keep in mind, if you change the number of neurons / filters chances are a shape mismatch might occur.
#
# For more information, see:
#
# * `Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. In: Neural Information Processing System Deep Learning Workshop (2015) <https://arxiv.org/abs/1503.02531>`_
#
# * `Romero, A., Ballas, N., Kahou, S.E., Chassang, A., Gatta, C., Bengio, Y.: Fitnets: Hints for thin deep nets. In: Proceedings of the International Conference on Learning Representations (2015) <https://arxiv.org/abs/1412.6550>`_
