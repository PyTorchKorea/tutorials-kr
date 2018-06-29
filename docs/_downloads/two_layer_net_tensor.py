# -*- coding: utf-8 -*-
"""
PyTorch: Tensor
----------------

하나의 은닉 계층과 편향(Bias)이 없는 완전히 연결된 ReLU 신경망에 유클리드
거리(Euclidean Distance)의 제곱을 최소화하여 x로부터 y를 예측하도록 학습하겠습니다.

PyTorch tensor를 사용하여 수동으로 순전파, 손실(loss), 그리고 역전파 연산을 하는
것을 구현해보겠습니다.

PyTorch Tensor는 기본적으로 NumPy 배열과 동일합니다; 딥러닝이나 연산 그래프
(Computational Graph), 변화도(Gradient)는 알지 못하며 임의의 숫자 계산에 사용하는
일반적인 N차원 배열입니다.

NumPy 배열과 PyTorch Tensor의 가장 큰 차이점은 PyTorch Tensor는 CPU나 GPU 어디서든
실행이 가능하다는 점입니다. GPU에서 연산을 하기 위해서는, Tensor를 CUDA 자료형으로
변환(Cast)해주기만 하면 됩니다.
"""

import torch


dtype = torch.float
device = torch.device("cpu")
# dtype = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요.

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉 계층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위의 입력과 출력 데이터를 생성합니다.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 무작위로 가중치를 초기화합니다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 순전파 단계: 예측값 y를 계산합니다.
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 손실(loss)을 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 경사하강법(Gradient Descent)를 사용하여 가중치를 갱신합니다.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
