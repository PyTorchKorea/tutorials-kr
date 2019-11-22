# -*- coding: utf-8 -*-
"""
PyTorch: 사용자 정의 nn Module
-------------------------------

하나의 은닉층(hidden layer)과 편향(bias)이 없는 완전히 연결된 ReLU 신경망을,
유클리드 거리(Euclidean distance) 제곱을 최소화하는 식으로 x로부터 y를 예측하도록
학습하겠습니다.

이번에는 사용자 정의 Module의 서브클래스로 모델을 정의합니다. 기존 Module의 간단한
구성보다 더 복잡한 모델을 원한다면, 이 방법으로 모델을 정의하면 됩니다.
"""
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        생성자에서 2개의 nn.Linear 모듈을 생성하고, 멤버 변수로 지정합니다.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        순전파 함수에서는 입력 데이터의 Tensor를 받고 출력 데이터의 Tensor를
        반환해야 합니다. Tensor 상의 임의의 연산자뿐만 아니라 생성자에서 정의한
        Module도 사용할 수 있습니다.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 앞에서 정의한 클래스를 생성하여 모델을 구성합니다.
model = TwoLayerNet(D_in, H, D_out)

# 손실 함수와 Optimizer를 만듭니다. SGD 생성자에 model.parameters()를 호출하면
# 모델의 멤버인 2개의 nn.Linear 모듈의 학습 가능한 매개변수들이 포함됩니다.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # 순전파 단계: 모델에 x를 전달하여 예상되는 y 값을 계산합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력합니다.
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
