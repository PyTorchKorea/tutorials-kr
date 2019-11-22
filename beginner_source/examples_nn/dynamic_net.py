# -*- coding: utf-8 -*-
"""
PyTorch: 제어 흐름(Control Flow) + 가중치 공유(Weight Sharing)
---------------------------------------------------------------

PyTorch 동적 그래프의 강력함을 보여주기 위해, 매우 이상한 모델을 구현해보겠습니다:
각 순전파 단계에서 많은 은닉 계층을 갖는 완전히 연결(fully-connected)된 ReLU
신경망이 무작위로 0 ~ 3 사이의 숫자를 선택하고, 가장 안쪽(innermost)의 은닉층들을
계산하기 위해 동일한 가중치를 여러 번 재사용합니다.
"""
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        생성자에서 순전파 단계에서 사용할 3개의 nn.Linear 인스턴스를 생성합니다.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        모델의 순전파 단계에서, 무작위로 0, 1, 2 또는 3 중에 하나를 선택하고
        은닉층을 계산하기 위해 여러번 사용한 middle_linear Module을 재사용합니다.

        각 순전파 단계는 동적 연산 그래프를 구성하기 때문에, 모델의 순전파 단계를
        정의할 때 반복문이나 조건문과 같은 일반적인 Python 제어 흐름 연산자를 사용할
        수 있습니다.

        여기에서 연산 그래프를 정의할 때 동일 Module을 여러번 재사용하는 것이
        완벽히 안전하다는 것을 알 수 있습니다. 이것이 각 Module을 한 번씩만 사용할
        수 있었던 Lua Torch보다 크게 개선된 부분입니다.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 앞서 정의한 클래스를 생성(instantiating)하여 모델을 구성합니다.
model = DynamicNet(D_in, H, D_out)

# 손실함수와 Optimizer를 만듭니다. 이 이상한 모델을 순수한 확률적 경사 하강법
# (stochastic gradient decent)으로 학습하는 것은 어려우므로, 모멘텀(momentum)을
# 사용합니다.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
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
