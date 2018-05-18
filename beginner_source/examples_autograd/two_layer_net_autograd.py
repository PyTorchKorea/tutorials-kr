# -*- coding: utf-8 -*-
"""
PyTorch: Tensor와 autograd
-----------------------------

하나의 은닉 계층(Hidden Layer)과 편향(Bias)이 없는 완전히 연결된 ReLU 신경망에
유클리드 거리(Euclidean Distance)의 제곱을 최소화하여 x로부터 y를 예측하도록
학습하겠습니다.

이 구현에서는 PyTorch Tensor 연산을 사용하여 순전파를 계산하고, PyTorch autograd를
사용하여 변화도(Gradient)를 계산하는 것을 구현하겠습니다.


PyTorch Tensor는 연산 그래프에서 노드(Node)로 표현(represent)됩니다. 만약 ``x`` 가
``x.requires_grad=True`` 인 Tensor라면 ``x.grad`` 는 어떤 스칼라 값에 대해 ``x`` 의
변화도(gradient)를 갖는 또 다른 Tensor입니다.
"""
import torch

dtype = torch.float
device = torch.device("cpu")
# dtype = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요.

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉 계층의 차원이며, D_out은 출력 차원입니다:
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장(Hold)하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
# requires_grade=False로 설정하여 역전파 중에 이 Tensor들에 대한 변화도를 계산할
# 필요가 없음을 나타냅니다.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 가중치를 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
# requires_grad=True로 설정하여 역전파 중에 이 Tensor들에 대한
# 변화도를 계산할 필요가 있음을 나타냅니다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 순전파 단계: Tensor 연산을 사용하여 y 값을 예측합니다. 이는 Tensor를 사용한
    # 순전파 단계와 완전히 동일하지만, 역전파 단계를 별도로 구현하지 않기 위해 중간
    # 값들(Intermediate Value)에 대한 참조(Reference)를 갖고 있을 필요가 없습니다.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Tensor 연산을 사용하여 손실을 계산하고 출력합니다.
    # loss는 (1,) 모양을 갖는 Variable이며, loss.data는 (1,) 모양의 Tensor입니다;
    # loss.data[0]은 손실(loss)의 스칼라 값입니다.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # autograde를 사용하여 역전파 단계를 계산합니다. 이는 requires_grad=True를
    # 갖는 모든 Tensor에 대한 손실의 변화도를 계산합니다. 이후 w1.grad와 w2.grad는
    # w1과 w2 각각에 대한 손실의 변화도를 갖는 Tensor가 됩니다.
    loss.backward()

    # 경사하강법(Gradient Descent)을 사용하여 가중치를 수동으로 갱신합니다.
    # 가중치들이 requires_grad=True 이기 때문에 torch.no_grad() 로 감싸지만,
    # autograd 내에서 이를 추적할 필요는 없습니다.
    # 다른 방법은 weight.data 및 weight.grad.data 를 조작(Operate)하는 방법입니다.
    # tensor.data 가 tensor의 저장공간(Storage)을 공유하기는 하지만, 이력을
    # 추적하지 않는다는 것을 기억하십시오.
    # 또한, 이것을 달성하기 위해 torch.optim.SGD 를 사용할 수도 있습니다.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
        w1.grad.zero_()
        w2.grad.zero_()
