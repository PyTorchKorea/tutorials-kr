"""
PyTorch의 Shape들에 대한 추론
=================================
번역: `이영섭 <https://github.com/0seob>`_

일반적으로 PyTorch로 모델을 작성할 때 특정 계층의 매개변수는 이전 계층의 출력 shape에 따라 달라집니다.
예를 들어, ``nn.Linear`` 계층의 ``in_features`` 는 입력의 ``size(-1)`` 와 일치해야 합니다.
몇몇 계층의 경우, shape 계산은 합성곱 연산과 같은 복잡한 방정식을 포함합니다.

이를 랜덤한 입력으로 순전파(forward pass)를 실행하여 해결할 수 있지만, 이는 메모리와 컴퓨팅 파워를 낭비합니다.

대신에 ``meta`` 디바이스를 활용한다면 데이터를 구체화하지 않고도 계층의 출력 shape을 결정할 수 있습니다.
"""

import torch
import timeit

t = torch.rand(2, 3, 10, 10, device="meta")
conv = torch.nn.Conv2d(3, 5, 2, device="meta")
start = timeit.default_timer()
out = conv(t)
end = timeit.default_timer()

print(out)
print(f"Time taken: {end-start}")


##########################################################################
# 데이터가 구체화되지 않기 때문에 임의로 큰 입력을 전달해도 shape 계산에 소요되는 시간이 
# 크게 변경되지는 않습니다.

t_large = torch.rand(2**10, 3, 2**16, 2**16, device="meta")
start = timeit.default_timer()
out = conv(t_large)
end = timeit.default_timer()

print(out)
print(f"Time taken: {end-start}")


######################################################
# 다음과 같은 임의의 네트워크를 가정합니다:

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화 합니다.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


###############################################################################
# 각각의 계층에 출력의 shape을 인쇄하는 forward hook을 등록하여 네트워크의 
# 중간 shape을 확인할 수 있습니다.

def fw_hook(module, input, output):
    print(f"Shape of output to {module} is {output.shape}.")


# torch.device context manager(with 구문) 내부에서 생성된 모든 tensor는 
# meta 디바이스 내부에 존재합니다.
with torch.device("meta"):
    net = Net()
    inp = torch.randn((1024, 3, 32, 32))

for name, layer in net.named_modules():
    layer.register_forward_hook(fw_hook)

out = net(inp)
