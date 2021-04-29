# -*- coding: utf-8 -*-
"""
파이토치(PyTorch): 텐서(Tensor)
--------------------------------

:math:`y=\sin(x)` 을 예측할 수 있도록, :math:`-\pi` 부터 :math:`pi` 까지
유클리드 거리(Euclidean distance)를 최소화하도록 3차 다항식을 학습합니다.

이 구현은 PyTorch 텐서를 사용하여 순전파 단계와 손실(loss), 역전파 단계를 직접 계산합니다.

PyTorch 텐서는 기본적으로 NumPy 배열과 동일하게 딥러닝이나 변화도(gradient), 
연산 그래프(computational graph)는 알지 못하며, 일반적인 n-차원 배열로 임의의 수치 연산에 사용됩니다.

NumPy 배열과 PyTorch 텐서의 가장 큰 차이점은 PyTorch 텐서는 CPU 및 GPU에서 실행될 수 있다는 것입니다.
GPU에서 연산을 실행하기 위해서는 텐서를 cuda 데이터형(datatype)으로 변환(cast)해주기만 하면 됩니다.
"""

import torch
import math


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요

# 무작위로 입력과 출력 데이터를 생성합니다
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 무작위로 가중치를 초기화합니다
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계: 예측값 y를 계산합니다
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 손실(loss)을 계산하고 출력합니다
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # 손실에 따른 a, b, c, d의 변화도(gradient)를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 가중치를 갱신합니다.
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
