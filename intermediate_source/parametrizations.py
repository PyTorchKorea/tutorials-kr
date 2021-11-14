# -*- coding: utf-8 -*-

"""
매개변수화 튜토리얼
=========================
**저자**: `Mario Lezcano <https://github.com/lezcano>`_

딥 러닝 모델을 정규화하는 것은 놀랍도록 어려운 작업입니다.
패널티 방법과 같은 고전적인 기술은 최적화되는 기능의 복잡성으로 인해 심층 모델에 적용할 때 종종 부족합니다.
이는 조건이 나쁜 모델로 작업할 때 특히 문제가 됩니다.
예를 들어 긴 시퀀스와 GAN에 대해 훈련된 RNN이 있습니다. 
이러한 모델을 정규화하고 수렴을 개선하기 위해 최근 몇 년 동안 많은 기술이 제안되었습니다.
순환 모델에서 RNN이 잘 조절되도록 순환 커널의 특이값을 제어하는 것이 제안되었습니다. 
이것은 예를 들어 순환 커널 
`직교 <https://en.wikipedia.org/wiki/Orthogonal_matrix>`_
을 만들어서 달성할 수 있습니다.
순환 모델을 정규화하는 또 다른 방법은 다음을 활용한 방법입니다 : 
"`비중 정규화 <https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html>`_".
이 접근 방식은 매개변수 학습을 규범 학습과 분리할 것을 제안합니다.
그렇게 하기 위해 매개변수는
`Frobenius norm <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`_
으로 나누어지고 해당 표준을 인코딩하는 별도의 매개변수가 학습됩니다.

"`spectral normalization <https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html>`_" 라는 이름으로 GAN에 대해 유사한 정규화가 제안되었습니다.
이 방법은 매개변수를 Frobenius 표준이 아닌 스펙트럼 표준 
 `spectral norm <https://en.wikipedia.org/wiki/Matrix_norm#Special_cases>`_,
으로 나누어 네트워크의 Lipschitz 상수를 제어합니다.

이러한 모든 방법에는 공통 패턴이 있습니다. 모든 방법은 매개변수를 사용하기 전에 적절한 방식으로 변환합니다.
첫 번째 경우에는 행렬을 직교 행렬에 매핑하는 함수를 사용하여 직교하게 만듭니다.
가중치 및 스펙트럼 정규화의 경우 원래 매개변수를 표준으로 나눕니다.

보다 일반적으로, 이러한 모든 예는 매개변수에 추가 구조를 추가하는 함수를 사용합니다.
즉, 매개변수를 제한하는 기능을 사용합니다.

이 자습서에서는 이 패턴을 구현하고 사용하여 모델에 제약 조건을 적용하는 방법을 배웁니다.그렇게 하는 것은 자신만의 ``nn.Module``를 작성하는 것만큼 쉽습니다.

요구사항: ``torch>=1.9.0``

수동으로 매개변수화 구현
-------------------------------------

가중치 ``X`` 로 ``X = Xᵀ`` 라는 대칭 가중치를 가진 정사각형 선형 레이어를 갖고 싶다고 가정합니다.
그렇게 하는 한 가지 방법은 행렬의 위쪽 삼각형 부분을 아래쪽 삼각형 부분으로 복사하는 것입니다
"""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)

X = torch.rand(3, 3)
A = symmetric(X)
assert torch.allclose(A, A.T)  # A 는 대칭이다.
print(A)                       # 빠른 육안 

###############################################################################
# 그런 다음 이 아이디어를 사용하여 대칭 가중치가 있는 선형 레이어를 구현할 수 있습니다.
class LinearSymmetric(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(n_features, n_features))

    def forward(self, x):
        A = symmetric(self.weight)
        return x @ A

###############################################################################
# 그런 다음 레이어를 일반 선형 레이어로 사용할 수 있습니다.
layer = LinearSymmetric(3)
out = layer(torch.rand(8, 3))

###############################################################################
# 이 구현은 정확하고 독립적이지만 여러 문제를 나타냅니다.:
#
# 1) 레이어를 다시 구현합니다. 선형 레이어를 ``x @ A`` 로 구현해야 했습니다. 
#    이것은 선형 레이어에서는 그다지 문제가 되지 않지만 CNN이나 Transformer를 다시 구현해야 한다고 상상해보세요...
# 2) 레이어와 매개변수화를 분리하지 않습니다. 
#    매개변수화가 더 어렵다면 사용하려는 각 계층에 대한 코드를 다시 작성해야 합니다.
# 3) 레이어를 사용할 때마다 매개변수화를 다시 계산합니다. 
#    만약 순방향 패스 동안 레이어를 여러 번 사용하면(RNN의 반복 커널을 상상해 보세요) 레이어가 호출될 때마다 동일한 ``A``를 계산합니다.
#
# 매개변수화 소개
# --------------------------------
#
# 매개변수화는 이러한 모든 문제는 물론 다른 문제도 해결할 수 있습니다.
#
# ``torch.nn.utils.parametrize`` 를 사용하여 위의 코드를 다시 구현하는 것으로 시작해봅시다.
# 우리가 해야 할 유일한 일은 매개변수화를 일반 ``nn.Module``로 작성하는 것 뿐 입니다.
class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

###############################################################################
# 이것이 우리가 해야 할 전부입니다. 일단 이것이 있으면  
layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Symmetric())
# 을 수행하여 일반 레이어를 대칭 레이어로 변환할 수 있습니다.

###############################################################################
# 이제 선형 레이어의 행렬은 대칭입니다.
A = layer.weight
assert torch.allclose(A, A.T)  # A is symmetric
print(A)                       # Quick visual check

###############################################################################
# 다른 레이어에서도 동일한 작업을 수행할 수 있습니다. 예를 들어 `skew-symmetric <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_ 
# 커널을 사용하여 CNN을 만들 수 있습니다.
# 우리는 유사한 매개변수화를 사용하여 기호가 반대인 위쪽 삼각형 부분을
# 아래쪽 삼각형 부분으로 복사합니다.
class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)


cnn = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3)
parametrize.register_parametrization(cnn, "weight", Skew())
# Print a few kernels
print(cnn.weight[0, 1])
print(cnn.weight[2, 2])

###############################################################################
# 매개변수화된 모듈 검사
# --------------------------------
#
# 모듈이 매개변수화되면 모듈이 세 가지 방식으로 변경되었음을 알 수 있습니다.:
#
# 1) ``model.weight`` 는 이제 속성입니다.
#
# 2) 이것은 새로운 ``module.parametrizations`` 속성을 가지고 있습니다.
#
# 3) 매개변수화되지 않은 가중치가 ``module.parametrizations.weight.original`` 로 이동되었습니다.
#
# |
#  ``weight`` 를 매개변수화하면, ``layer.weight`` 는
# `Python 속성 <https://docs.python.org/3/library/functions.html#property>`_ 로 바뀝니다.
# 이 속성은  ``layer.weight`` 을 요청할 때마다 ``LinearSymmetric`` 에서 구현했던 것 처럼
# ``parametrization(weight)`` 을 구현합니다..
#
# 등록돈 매개변수들은 모듈 내의 ``parametrizations`` 속성 아래로 저장됩니다.
layer = nn.Linear(3, 3)
print(f"Unparametrized:\n{layer}")
parametrize.register_parametrization(layer, "weight", Symmetric())
print(f"\nParametrized:\n{layer}")

###############################################################################
# 이 ``parametrizations`` 속성은 ``nn.ModuleDict``이며 다음과 같이 액세스할 수 있습니다.
print(layer.parametrizations)
print(layer.parametrizations.weight)

###############################################################################
# 이 ``nn.ModuleDict``의 각 요소는 ``nn.Sequential``처럼 작동하는 ``ParametrizationList``입니다.
# 이 목록을 사용하면 매개변수화를 하나의 가중치로 연결할 수 있습니다.
# 이것은 list 이기 때문에 그것을 인덱싱하는 매개변수화에 접근할 수 있습니다.
# 여기에서 ``Symmetric`` 매개변수화가 수행됩니다.
print(layer.parametrizations.weight[0])

###############################################################################
# 우리가 알아차린 또 다른 점은 매개변수를 출력하면
# ``weight`` 매개변수가 이동되었음을 알 수 있다는 것입니다.
print(dict(layer.named_parameters()))

###############################################################################
# 이것은 이제 ``layer.parametrizations.weight.original`` 아래에 있습니다.
print(layer.parametrizations.weight.original)

###############################################################################
# 이 세 가지 작은 차이점 외에도 매개 변수화는
# 수동 구현과 정확히 동일합니다.
symmetric = Symmetric()
weight_orig = layer.parametrizations.weight.original
print(torch.dist(layer.weight, symmetric(weight_orig)))

###############################################################################
# 매개변수화는 일급 객체입니다.
# -----------------------------------------
#
# ``layer.parametrizations`` 는 ``nn.ModuleList`` 이므로, 매개변수가 원래
# 모듈의 하위 모듈로 제대로 등록되었음을 의미합니다. 
# 따라서 모듈에 매개변수를 등록하는 것과 동일한 규칙이 매개변수화를 등록하는 데 적용됩니다.
# 예를 들어 매개변수에 매개변수가 있는 경우, ``model = model.cuda()`  
# 를 호출할 때 매개변수가 CPU에서 CUDA로 이동됩니다.
#
# 매개변수화 값 캐싱
# --------------------------------------
#
# 매개변수화는 컨텍스트 관리자 ``parametrize.cached()`` 를 통해
# 내장된 캐싱 시스템과 함께 제공됩니다
class NoisyParametrization(nn.Module):
    def forward(self, X):
        print("Computing the Parametrization")
        return X

layer = nn.Linear(4, 4)
parametrize.register_parametrization(layer, "weight", NoisyParametrization())
print("Here, layer.weight is recomputed every time we call it")
foo = layer.weight + layer.weight.T
bar = layer.weight.sum()
with parametrize.cached():
    print("Here, it is computed just the first time layer.weight is called")
    foo = layer.weight + layer.weight.T
    bar = layer.weight.sum()

###############################################################################
# 매개변수화 연결
# ------------------------------
#
# 두 개의 매개변수화를 연결하는 것은 동일한 텐서에 등록하는 것만큼 쉽습니다.
# 우리는 이것을 사용하여 더 단순한 것에서 더 복잡한 매개변수화를 생성할 수 있습니다.
# 예를 들어 `Cayley map <https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map>`_ 는
# 비대칭 행렬을 양수 행렬의 직교 행렬에 매핑합니다. 
# 직교 가중치가 있는 레이어를 얻기 위해 Cayley 맵을 구현하는 매개변수화와
#  ``Skew`` 를 연결할 수 있습니다.
class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        return torch.solve(self.Id + X, self.Id - X).solution

layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
parametrize.register_parametrization(layer, "weight", CayleyMap(3))
X = layer.weight
print(torch.dist(X.T @ X, torch.eye(3)))  # X는 직교

###############################################################################
# 이것은 매개변수화된 모듈을 제거하거나 매개변수화를 재사용하는 데에도 사용할 수 있습니다.
# 예를 들어, 행렬 지수는 대칭 행렬을 대칭 양수(SPD) 행렬에 매핑하지만 
# 행렬 지수는 비대칭 행렬도 직교 행렬에 매핑합니다. 이 두 가지 사실을 사용하여
# 이전에 매개변수화를 재사용할 수 있습니다.
class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", MatrixExponential())
X = layer_orthogonal.weight
print(torch.dist(X.T @ X, torch.eye(3)))         # X는 직교

layer_spd = nn.Linear(3, 3)
parametrize.register_parametrization(layer_spd, "weight", Symmetric())
parametrize.register_parametrization(layer_spd, "weight", MatrixExponential())
X = layer_spd.weight
print(torch.dist(X, X.T))                        # X 는 대칭
print((torch.symeig(X).eigenvalues > 0.).all())  # X 는 양수

###############################################################################
# 매개변수화 초기화
# ----------------------------
#
# 매개변수화에는 매개변수화를 초기화하는 메커니즘이 함께 제공됩니다.
# 만약 우리가 하단의 표시가 있는``right_inverse`` 메서드를 구현하면
#
# .. code-block:: python
#
#     def right_inverse(self, X: Tensor) -> Tensor
#
# 매개변수화된 텐서에 할당할 때 사용될 것입니다.
#
# L이것을 지원하기 위해 ``Skew`` 클래스의 구현을 업그레이드해봅시다.
class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)

    def right_inverse(self, A):
        # A가 비대칭 대칭이라고 가정합니다.
        # 우리는 앞으로 사용될 위쪽 삼각형 요소를 사용합니다.
        return A.triu(1)

###############################################################################
# 이제 ``Skew``로 매개변수화된 레이어를 초기화할 수 있습니다.
layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
X = torch.rand(3, 3)
X = X - X.T                             # X 는 이제 비대칭
layer.weight = X                        # layer.weight를 X로 초기화
print(torch.dist(layer.weight, X))      # layer.weight == X

###############################################################################
# 이 ``right_inverse`` 는 매개변수화를 연결할 때 예상대로 작동합니다.
# 이를 확인하기 위해 Cayley 매개변수화를 업그레이드하여 초기화도 지원하도록 합시다.
class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):
        # X는 비대칭이라고 가정
        # (I + X)(I - X)^{-1}
        return torch.solve(self.Id + X, self.Id - X).solution

    def right_inverse(self, A):
        # A는 직교로 가정
        # https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map 
        # (X - I)(X + I)^{-1}
        return torch.solve(X - self.Id, self.Id + X).solution

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", CayleyMap(3))
# 양의 행렬식이 있는 직교 행렬 샘플링
X = torch.empty(3, 3)
nn.init.orthogonal_(X)
if X.det() < 0.:
    X[0].neg_()
layer_orthogonal.weight = X
print(torch.dist(layer_orthogonal.weight, X))  # layer_orthogonal.weight == X

###############################################################################
# 이 초기화 단계는 다음과 같이 더 간결하게 작성할 수 있습니다.
layer_orthogonal.weight = nn.init.orthogonal_(layer_orthogonal.weight)

###############################################################################
# 이 메서드의 이름은 우리가 종종 ``forward(right_inverse(X)) == X`` 라고 예상한다는
# 사실에서 유래합니다. 이것은 ``X`` 값으로 초기화를 수행한 후 앞으로 ``X`` 값을
# 반환해야 한다는 것을 다시 작성하는 직접적인 방법입니다. 이 제약은 실제로 강력하게 시행되지 않습니다. 
# 사실, 때때로 이 관계를 완화하는 것이 흥미로울 수 있습니다.
# 예를 들어, 무작위 가지치기 방법의 다음 구현을 고려하십시오. :

class PruningParametrization(nn.Module):
    def __init__(self, X, p_drop=0.2):
        super().__init__()
        # sample zeros with probability p_drop
        mask = torch.full_like(X, 1.0 - p_drop)
        self.mask = torch.bernoulli(mask)

    def forward(self, X):
        return X * self.mask

    def right_inverse(self, A):
        return A

###############################################################################
# 이 경우 모든 행렬 A에 대해  ``forward(right_inverse(A)) == A``라는 것은 사실이 아닙니다.
# 이는 행렬 ``A`` 가 마스크와 동일한 위치에 0을 가질 때만 해당됩니다.
# 그렇더라도 텐서를 정리된 매개변수에 할당하면 실제로 텐서가 정리되는 것은 놀라운 일이 아닙니다.
layer = nn.Linear(3, 4)
X = torch.rand_like(layer.weight)
print(f"Initialization matrix:\n{X}")
parametrize.register_parametrization(layer, "weight", PruningParametrization(layer.weight))
layer.weight = X
print(f"\nInitialized weight:\n{layer.weight}")

###############################################################################
# 매개변수화 제거
# -------------------------
#
# ``parametrize.remove_parametrizations()``을 사용하여
#  모듈의 매개변수 또는 버퍼에서 모든 매개변수화를 제거할 수 있습니다.
layer = nn.Linear(3, 3)
print("Before:")
print(layer)
print(layer.weight)
parametrize.register_parametrization(layer, "weight", Skew())
print("\nParametrized:")
print(layer)
print(layer.weight)
parametrize.remove_parametrizations(layer, "weight")
print("\nAfter. Weight has skew-symmetric values but it is unconstrained:")
print(layer)
print(layer.weight)

###############################################################################
# 매개변수화를 제거할 때 플래그 ``leave_parametrized=False`` 를 설정하여 매개변수화된 버전 대신
# 원래 매개변수(예: ``layer.parametriations.weight.original``에 있는 매개변수)를 그대로 둘 수 있습니다.

layer = nn.Linear(3, 3)
print("Before:")
print(layer)
print(layer.weight)
parametrize.register_parametrization(layer, "weight", Skew())
print("\nParametrized:")
print(layer)
print(layer.weight)
parametrize.remove_parametrizations(layer, "weight", leave_parametrized=False)
print("\nAfter. Same as Before:")
print(layer)
print(layer.weight)
