# -*- coding: utf-8 -*-
"""
PyTorch: optim
--------------

하나의 은닉층(hidden layer)과 편향(bias)이 없는 완전히 연결된 ReLU 신경망을,
유클리드 거리(Euclidean distance) 제곱을 최소화하는 식으로 x로부터 y를 예측하도록
학습하겠습니다.

이번에는 PyTorch의 nn 패키지를 사용하여 신경망을 구현해보겠습니다.

지금까지 해왔던 것처럼 직접 모델의 가중치를 갱신하는 대신, optim 패키지를 사용하여
가중치를 갱신할 Optimizer를 정의합니다. optim 패키지는 일반적으로 딥러닝에 사용하는
SGD+momentum, RMSProp, Adam 등과 같은 다양한 최적화(Optimization) 알고리즘을
정의합니다.
"""
import torch

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# nn 패키지를 사용하여 모델과 손실 함수를 정의합니다.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# optim 패키지를 사용하여 모델의 가중치를 갱신할 Optimizer를 정의합니다.
# 여기서는 Adam을 사용하겠습니다; optim 패키지는 다른 다양한 최적화 알고리즘을
# 포함하고 있습니다. Adam 생성자의 첫번째 인자는 어떤 Tensor가 갱신되어야 하는지
# 알려줍니다.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # 순전파 단계: 모델에 x를 전달하여 예상되는 y 값을 계산합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력합니다.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계 전에, Optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인)
    # 갱신할 변수들에 대한 모든 변화도를 0으로 만듭니다. 이렇게 하는 이유는
    # 기본적으로 .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고)
    # 누적되기 때문입니다. 더 자세한 내용은 torch.autograd.backward에 대한 문서를
    # 참조하세요.
    optimizer.zero_grad()

    # 역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산합니다.
    loss.backward()

    # Optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
    optimizer.step()
