"""
변화도 시각화
=====================

**저자""
변화도 시각화
=====================

**저자:** `Justin Silver <https://github.com/j-silv>`__

이 튜토리얼은 신경망의 어떤 레이어에서든 변화도를 추출하고 시각화하는 방법을 설명합니다. 
어게 정보가 네트워크의 끝에서 우리가 원하는 매변수까지 흐르는지 점검함으로써 
우리는 학습 중 발생하는 '변화도 소실 또는 폭발 <https://arxiv.org/abs/1211.5063>`__과 같은 
문제를 디버깅할 수 있습니다.

시작하기 전에 'tensor와 그것들을 조작하는 방법
<https://docs.tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html>`__을 확실히 이해하세요.
기본적인 'autograd 작동법
<https://docs.tutorials.pytorch.kr/beginner/basics/autogradqs_tutorial.html>`__
을 알아두는 것 또한 유용합니다.

"""


######################################################################
# 설정
# -----
#
# 우선, `파이토치가 설치되었는지
# <https://pytorch.org/get-started/locally/>`__ 확인하고
# 필요한 라이브러리들을 import 하세요.
#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


######################################################################
# 다음으로, '배치 정규화 논문 <https://arxiv.org/abs/1502.03167>'__에서
# 설명된 아키텍처와 유사한, MNIST 데이터셋에 적합한 네트워크를 구축할 것입니다.
# 
#
# 변화도 시각화의 중요성을 설명하기 위해, 우리는 배치 정규화를 적용한
# 네트워크 버전(BatchNorm)과 적용하지 않은 버전을 각각 하나씩 생성할 것입니다.
# 우리는 배치 정규화가 '변화도 소실/폭발<https://arxiv.org/abs/1211.5063>'__을 
# 해결하는 데 매우 효과적인 기술임을 실험적으로 검증할 것입니다.
# 
# 
#
# 
# 우리가 사용하는 모델은 ``nn.Linear``, ``norm_layer``, 그리고 ``nn.Sigmoid``를
# 번갈아 사용하는 변경 가능한 개수의 반복되는 완전 연결 (fully-connected)레이어를 가지고 있습니다.
# 'norm_layer'는 배치 정규화가 활성화된 경우에는 
# `BatchNorm1d <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>`__
# 을 사용하고 그렇지 않은 경우에는
# `Identity <https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html>`__
# 변환을 사용할 것입니다.
#

def fc_layer(in_size, out_size, norm_layer):
    """Return a stack of linear->norm->sigmoid layers"""
    return nn.Sequential(nn.Linear(in_size, out_size), norm_layer(out_size), nn.Sigmoid())

class Net(nn.Module):
    """Define a network that has num_layers of linear->norm->sigmoid transformations"""
    def __init__(self, in_size=28*28, hidden_size=128,
                 out_size=10, num_layers=3, batchnorm=False):
        super().__init__()
        if batchnorm is False:
            norm_layer = nn.Identity
        else:
            norm_layer = nn.BatchNorm1d

        layers = []
        layers.append(fc_layer(in_size, hidden_size, norm_layer))

        for i in range(num_layers-1):
            layers.append(fc_layer(hidden_size, hidden_size, norm_layer))

        layers.append(nn.Linear(hidden_size, out_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.layers(x)


######################################################################
# 이제 우리는 더미 데이터(dummy data)를 준비하여 두 버전의 모델을 생성하고
# 옵티마이저를 초기화합니다.
#

# 더미 데이터(dummy data) 설정
x = torch.randn(10, 28, 28)
y = torch.randint(10, (10, ))

# 모델 초기화
model_bn = Net(batchnorm=True, num_layers=3)
model_nobn = Net(batchnorm=False, num_layers=3)

model_bn.train()
model_nobn.train()

optimizer_bn = optim.SGD(model_bn.parameters(), lr=0.01, momentum=0.9)
optimizer_nobn = optim.SGD(model_nobn.parameters(), lr=0.01, momentum=0.9)



######################################################################
# 우리는 내부 레이어 중 하나를 조사하여 배치 정규화가 하나의 모델에만 
# 적용되는지 확인할 수 있습니다.
#

print(model_bn.layers[0])
print(model_nobn.layers[0])


######################################################################
# 훅(hooks) 등록
# -----------------
#


######################################################################
# 우리가 ``nn.Module`` 안에 있는 모델의 논리(logic)과 상태를 래핑(wrap up)했기 때문에, 
# 모듈 코드를 직접 수정하지 않고 싶다면 중간 변화도에 접근하기 위한 새로운 방법이 필요합니다.
# 새로운 방법은 '훅 등록하기 <https://docs.pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution>`__'
# 를 통해 수행할 수 있습니다.
# 
#
# .. 경고::
#
#    tensor 자체에 'retain_grad()'를 사용하는 것보다 출력 tensor에 연결된 역전파 훅을 사용하는 것이 권장됩니다. "nn.Module" 인스턴스가 제자리 연산(in-place operation)을 수행하지 않는다면 모듈 훅을 직접 연결하는 대안(예: "register_full_backward_hook()``)도 있습니다. 더 자세한 정보는 `이 이슈 <https://github.com/pytorch/pytorch/issues/61519>`__를 참고해주세요.   
#
# 다음 코드는 훅을 정의하고 네트워크 계층(network layer)에 대한 묘사 명칭(descriptive name)을 수집합니다.
# 
#

# 인자를 전달할 수 있도록 파이썬 클로저를 위해 래퍼 함수가 사용된다는 점을 유의하세요.
# 

def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        """Forward pass hook which attaches backward pass hooks to intermediate tensors"""
        output.register_hook(hook_backward(module_name, grads))
    return hook

def hook_backward(module_name, grads):
    def hook(grad):
        """Backward pass hook which appends gradients"""
        grads.append((module_name, grad))
    return hook

def get_all_layers(model, hook_forward, hook_backward):
    """Register forward pass hook (which registers a backward hook) to model outputs

    Returns:
        - layers: a dict with keys as layer/module and values as layer/module names
                  e.g. layers[nn.Conv2d] = layer1.0.conv1
        - grads: a list of tuples with module name and tensor output gradient
                 e.g. grads[0] == (layer1.0.conv1, tensor.Torch(...))
    """
    layers = dict()
    grads = []
    for name, layer in model.named_modules():
        # skip Sequential and/or wrapper modules
        if any(layer.children()) is False:
            layers[layer] = name
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads

# 훅 등록
layers_bn, grads_bn = get_all_layers(model_bn, hook_forward, hook_backward)
layers_nobn, grads_nobn = get_all_layers(model_nobn, hook_forward, hook_backward)


######################################################################
# 학습 및 시각화
# --------------------------
#
# 이제 모델을 몇 에폭동안 학습시켜 보겠습니다:
#

epochs = 10

for epoch in range(epochs):

    # important to clear, because we append to
    # outputs everytime we do a forward pass
    grads_bn.clear()
    grads_nobn.clear()

    optimizer_bn.zero_grad()
    optimizer_nobn.zero_grad()

    y_pred_bn = model_bn(x)
    y_pred_nobn = model_nobn(x)

    loss_bn = F.cross_entropy(y_pred_bn, y)
    loss_nobn = F.cross_entropy(y_pred_nobn, y)

    loss_bn.backward()
    loss_nobn.backward()

    optimizer_bn.step()
    optimizer_nobn.step()


######################################################################
# 순방향 및 역방향 패스를 실행한 후, 모든 중간 tensor에 대한 변화도는
# ``grads_bn``와 ``grads_nobn``에 존재해야 합니다.
# 두 모델을 비교할 수 있도록 각 기울기 행렬의 평균 절댓값을 계산합니다.
# 
#

def get_grads(grads):
    layer_idx = []
    avg_grads = []
    for idx, (name, grad) in enumerate(grads):
        if grad is not None:
            avg_grad = grad.abs().mean()
            avg_grads.append(avg_grad)
            # idx is backwards since we appended in backward pass
            layer_idx.append(len(grads) - 1 - idx)
    return layer_idx, avg_grads

layer_idx_bn, avg_grads_bn = get_grads(grads_bn)
layer_idx_nobn, avg_grads_nobn = get_grads(grads_nobn)


######################################################################
# 우리는 이제 계산한 평균 변화도를 그래프로 나타내어 네트워크 깊이에 따라 
# 값이 어떻게 변하는지 확인할 수 있습니다.
# 우리가 배치 정규화를 적용하지 않으면 중간 기울기의 변화도가
# 급격하게 0으로 낮아지는 것을 알 수 있습니다.
# 반면, 배치 정규화를 적용한 모델은 중간 레이어에서 0이 아닌 기울기를 유지합니다.
#

fig, ax = plt.subplots()
ax.plot(layer_idx_bn, avg_grads_bn, label="With BatchNorm", marker="o")
ax.plot(layer_idx_nobn, avg_grads_nobn, label="Without BatchNorm", marker="x")
ax.set_xlabel("Layer depth")
ax.set_ylabel("Average gradient")
ax.set_title("Gradient flow")
ax.grid(True)
ax.legend()
plt.show()


######################################################################
# 결론
# ----------
# 이 튜토리얼에서는 'nn.Module' 클래스로 래핑된 신경망을 통해 변화도 흐름을 
# 시각화하는 방법을 설명하였습니다.
# 또한, 배치 정규화가 심층 신경망에서 발생하는 기울기 소실 문제를 완화하는 데
# 어떻게 도움이 되는지를 정성적으로 보여주었습니다.
# 
# 파이토치의 자동 미분 시스템 작동하는 방식을 추가로 학습하고 싶다면
# 아래 `참고 자료 <#references>`__를 확인하세요.
# 이 튜토리얼에 대한 피드백(개선 사항, 오타 수정 등)이 있다면 '파이토치 포럼<https://discuss.pytorch.org/>`__
# 그리고/또는 `이슈 트래커<https://github.com/pytorchkorea/tutorials-kr/issues>`__ 
# 를 통해 알려주시기 바랍니다.
#  
# 
#


######################################################################
# (선택 사항) 추가 연
# -------------------------------
#
# -  모델의 레이어 수(``num_layers``)를 늘려보고 
#    이것이 그래디언트 흐름 그래프에 어떤 영향을 미치는지 확인해보세요.
# -  평균 변화도 대신 평균 활성값을 시각화하려면 코드를 어떻게 수정해야 할까요?
#    (*힌트: hook_forward() 함수가 원시 tensor 출력에 접근할 수 있습니다*)
# -  변화도 소실 및 폭주 문제를 해결하기 위한 다른 방법에는 어떤 것들이 있을까요?
# 
#   
#


######################################################################
# 참고 문헌
# ----------
#
# -  `A Gentle Introduction to
#    torch.autograd <https://docs.tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html>`__
# -  `Automatic Differentiation with
#    torch.autograd <https://docs.tutorials.pytorch.kr/beginner/basics/autogradqs_tutorial>`__
# -  `Autograd
#    mechanics <https://docs.pytorch.org/docs/stable/notes/autograd.html>`__
# -  `Batch Normalization: Accelerating Deep Network Training by Reducing
#    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__
# -  `On the difficulty of training Recurrent Neural
#    Networks <https://arxiv.org/abs/1211.5063>`__
#
