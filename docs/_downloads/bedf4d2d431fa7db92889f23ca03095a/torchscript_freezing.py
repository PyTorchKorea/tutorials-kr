"""
TorchScript로 모델 동결하기
=============================

번역 : `김지호 <https://github.com/jiho3004/>`_

이 튜토리얼에서는, TorchScript로 *모델 동결* 하는 문법을 소개합니다.
동결은 파이토치 모듈의 매개변수와 속성 값들을 TorchScript 내부 표현으로 인라이닝(inlining)하는 과정입니다.
매개변수와 속성 값들은 최종 값으로 처리되며 동결된 모듈에서 수정될 수 없습니다.

기본 문법
------------

모델 동결은 아래 API를 사용하여 호출할 수 있습니다:

 ``torch.jit.freeze(mod : ScriptModule, names : str[]) -> ScriptModule``

입력 모듈은 스크립팅(scripting) 혹은 추적(tracing)을 사용한 결과입니다.
`TorchScript 소개 튜토리얼 <https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html>`_
을 참조하세요.

다음으로, 예제를 통해 동결이 어떤 방식으로 동작하는지 확인합니다:
"""

import torch, time

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

    @torch.jit.export
    def version(self):
        return 1.0

net = torch.jit.script(Net())
fnet = torch.jit.freeze(net)

print(net.conv1.weight.size())
print(net.conv1.bias)

try:
    print(fnet.conv1.bias)
    # 예외 처리가 없을 시 'conv1' 이라는 이름과 함께 다음을 출력합니다.
    # RuntimeError: __torch__.z.___torch_mangle_3.Net does not have a field
except RuntimeError:
    print("field 'conv1' is inlined. It does not exist in 'fnet'")

try:
    fnet.version()
    # 예외 처리가 없을 시 'version' 이라는 이름과 함께 다음을 출력합니다.
    # RuntimeError: __torch__.z.___torch_mangle_3.Net does not have a field
except RuntimeError:
    print("method 'version' is not deleted in fnet. Only 'forward' is preserved")

fnet2 = torch.jit.freeze(net, ["version"])

print(fnet2.version())

B=1
warmup = 1
iter = 1000
input = torch.rand(B, 1,28, 28)

start = time.time()
for i in range(warmup):
    net(input)
end = time.time()
print("Scripted - Warm up time: {0:7.4f}".format(end-start), flush=True)

start = time.time()
for i in range(warmup):
    fnet(input)
end = time.time()
print("Frozen   - Warm up time: {0:7.4f}".format(end-start), flush=True)

start = time.time()
for i in range(iter):
    input = torch.rand(B, 1,28, 28)
    net(input)
end = time.time()
print("Scripted - Inference: {0:5.2f}".format(end-start), flush=True)

start = time.time()
for i in range(iter):
    input = torch.rand(B, 1,28, 28)
    fnet2(input)
end = time.time()
print("Frozen    - Inference time: {0:5.2f}".format(end-start), flush =True)

###############################################################
# 개인 머신에서 시간을 측정한 결과입니다:
#
# * Scripted - Warm up time:  0.0107
# * Frozen   - Warm up time:  0.0048
# * Scripted - Inference:  1.35
# * Frozen   - Inference time:  1.17

###############################################################
# 이 예제에서, 워밍업 시간은 최초 두 번 실행할 때 측정합니다.
# 동결된 모델이 스크립트된 모델보다 50% 더 빠릅니다.
# 보다 복잡한 모델에서는 워밍업 시간이 더욱 빨라집니다.
# 최초 두 번의 실행을 초기화할 때 TorchScript가 해야 할 일의 일부를 동결이 하고 있기 때문에 속도 개선이 일어납니다.
#
# 추론 시간은 모델이 워밍업되고 난 뒤, 추론 시 실행 시간을 측정합니다.
# 실행 시간에 많은 편차가 있기는 하지만, 대개 동결된 모델이 스크립트된 모델보다 약 15% 더 빠릅니다.
# 실행 시간은 tensor 연산에 의해 지배되기 때문에 입력의 크기가 더 커지면 속도 개선 정도는 더 작아집니다.

###############################################################
# 결론
# -----------
#
# 이 튜토리얼에서는 모델 동결에 대해 배웠습니다.
# 동결은 추론 시 모델 최적화를 할 수 있는 유용한 기법이며 TorchScript 워밍업 시간을 크게 줄입니다.
