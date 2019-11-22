# -*- coding: utf-8 -*-
"""
PyTorch: nn
-----------

하나의 은닉층(hidden layer)과 편향(bias)이 없는 완전히 연결된 ReLU 신경망을,
유클리드 거리(Euclidean distance) 제곱을 최소화하는 식으로 x로부터 y를 예측하도록
학습하겠습니다.

이번에는 PyTorch의 nn 패키지를 사용하여 신경망을 구현하겠습니다.
PyTorch autograd는 연산 그래프를 정의하고 변화도를 계산하는 것을 손쉽게 만들어주지만,
autograd 그 자체만으로는 복잡한 신경망을 정의하기에는 너무 저수준(low-level)일 수
있습니다; 이것이 nn 패키지가 필요한 이유입니다. nn 패키지는 Module의 집합을
정의하는데, 이는 입력으로부터 출력을 생성하고 학습 가능한 가중치를 갖는
신경망 계층(neural network layer)이라고 생각할 수 있습니다.
"""
import torch

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# nn 패키지를 사용하여 모델을 순차적 계층(sequence of layers)으로 정의합니다.
# nn.Sequential은 다른 Module들을 포함하는 Module로, 그 Module들을 순차적으로
# 적용하여 출력을 생성합니다. 각각의 Linear Module은 선형 함수를 사용하여
# 입력으로부터 출력을 계산하고, 내부 Tensor에 가중치와 편향을 저장합니다.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# 또한 nn 패키지에는 널리 사용하는 손실 함수들에 대한 정의도 포함하고 있습니다;
# 여기에서는 평균 제곱 오차(MSE; Mean Squared Error)를 손실 함수로 사용하겠습니다.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # 순전파 단계: 모델에 x를 전달하여 예상되는 y 값을 계산합니다. Module 객체는
    # __call__ 연산자를 덮어써(override) 함수처럼 호출할 수 있게 합니다.
    # 이렇게 함으로써 입력 데이터의 Tensor를 Module에 전달하여 출력 데이터의
    # Tensor를 생성합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력합니다. 예측한 y와 정답인 y를 갖는 Tensor들을 전달하고,
    # 손실 함수는 손실 값을 갖는 Tensor를 반환합니다.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계를 실행하기 전에 변화도를 0으로 만듭니다.
    model.zero_grad()

    # 역전파 단계: 모델의 학습 가능한 모든 매개변수에 대해 손실의 변화도를
    # 계산합니다. 내부적으로 각 Module의 매개변수는 requires_grad=True 일 때
    # Tensor 내에 저장되므로, 이 호출은 모든 모델의 모든 학습 가능한 매개변수의
    # 변화도를 계산하게 됩니다.
    loss.backward()

    # 경사하강법(gradient descent)를 사용하여 가중치를 갱신합니다. 각 매개변수는
    # Tensor이므로 이전에 했던 것과 같이 변화도에 접근할 수 있습니다.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
