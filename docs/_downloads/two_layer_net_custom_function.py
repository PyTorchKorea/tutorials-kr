# -*- coding: utf-8 -*-
"""
PyTorch: 새 autograd 함수 정의하기
-----------------------------------

하나의 은닉 계층(Hidden Layer)과 편향(Bias)이 없는 완전히 연결된 ReLU 신경망에
유클리드 거리(Euclidean Distance)의 제곱을 최소화하여 x로부터 y를 예측하도록
학습하겠습니다.

PyTorch Variable 연산을 사용하여 순전파를 계산하고, PyTorch autograd를 사용하여
변화도(Gradient)를 계산하는 것을 구현하겠습니다.

여기서는 사용자 정의 autograd 함수를 구현하여 ReLU 기능(function)을 수행하도록
하겠습니다.
"""
import torch


class MyReLU(torch.autograd.Function):
    """
    torch.autograd.Function을 상속받아 사용자 정의 autograd 함수를 구현하고,
    Tensor 연산을 하는 순전파와 역전파 단계를 구현하겠습니다.
    """

    @staticmethod
    def forward(ctx, input):
        """
        순전파 단계에서는 입력을 갖는 Tensor를 받아 출력을 갖는 Tensor를 반환합니다.
        ctx는 역전파 연산을 위한 정보를 저장하기 위해 사용하는 Context Object입니다.
        ctx.save_for_backward method를 사용하여 역전파 단계에서 사용할 어떠한
        객체(object)도 저장(cache)해 둘 수 있습니다.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        역전파 단계에서는 출력에 대한 손실의 변화도를 갖는 Tensor를 받고, 입력에
        대한 손실의 변화도를 계산합니다.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# dtype = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요.

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉 계층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장(Hold)하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 가중치를 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 사용자 정의 함수를 적용하기 위해 Function.apply method를 사용합니다.
    # 이를 'relu'라고 이름(alias) 붙였습니다.
    relu = MyReLU.apply

    # 순전파 단계: Tensor 연산을 사용하여 y 값을 예측합니다;
    # 사용자 정의 autograd 연산을 사용하여 ReLU를 계산합니다.
    y_pred = relu(x.mm(w1)).mm(w2)

    # 손실(loss)을 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # autograde를 사용하여 역전파 단계를 계산합니다.
    loss.backward()

    # 경사하강법(Gradient Descent)을 사용하여 가중치를 갱신합니다.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
        w1.grad.zero_()
        w2.grad.zero_()
