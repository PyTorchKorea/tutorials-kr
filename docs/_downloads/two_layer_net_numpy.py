# -*- coding: utf-8 -*-
"""
준비 운동: NumPy
-----------------

하나의 은닉 계층과 편향(Bias)이 없는 완전히 연결된 ReLU 신경망에 유클리드
오차(Euclidean Error)를 사용하여 x로부터 y를 예측하도록 학습하겠습니다.

NumPy를 사용하여 수동으로 순전파, 손실(loss), 그리고 역전파 연산을 하는 것을
구현해보겠습니다.

NumPy 배열은 일반적은 N차원 배열입니다; 딥러닝이나 변화도(Gradient), 연산
그래프(Computational Graph)는 알지 못하며 일반적인 수치 연산을 수행합니다.
"""
import numpy as np

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉 계층의 차원이며, D_out은 출력 차원입니다:
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위의 입력과 출력 데이터를 생성합니다.
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 무작위로 가중치를 초기화합니다.
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # 순전파 단계: 예측값 y를 계산합니다.
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 손실(loss)을 계산하고 출력합니다.
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 가중치를 갱신합니다.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
