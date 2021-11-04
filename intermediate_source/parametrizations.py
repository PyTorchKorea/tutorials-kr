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
assert torch.allclose(A, A.T)  # A is symmetric
print(A)                       # Quick visual check

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
# Now, the matrix of the linear layer is symmetric
A = layer.weight
assert torch.allclose(A, A.T)  # A is symmetric
print(A)                       # Quick visual check

###############################################################################
# We can do the same thing with any other layer. For example, we can create a CNN with
# `skew-symmetric <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_ kernels.
# We use a similar parametrization, copying the upper-triangular part with signs
# reversed into the lower-triangular part
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
# Inspecting a parametrized module
# --------------------------------
#
# When a module is parametrized, we find that the module has changed in three ways:
#
# 1) ``model.weight`` is now a property
#
# 2) It has a new ``module.parametrizations`` attribute
#
# 3) The unparametrized weight has been moved to ``module.parametrizations.weight.original``
#
# |
# After parametrizing ``weight``, ``layer.weight`` is turned into a
# `Python property <https://docs.python.org/3/library/functions.html#property>`_.
# This property computes ``parametrization(weight)`` every time we request ``layer.weight``
# just as we did in our implementation of ``LinearSymmetric`` above.
#
# Registered parametrizations are stored under a ``parametrizations`` attribute within the module.
layer = nn.Linear(3, 3)
print(f"Unparametrized:\n{layer}")
parametrize.register_parametrization(layer, "weight", Symmetric())
print(f"\nParametrized:\n{layer}")

###############################################################################
# This ``parametrizations`` attribute is an ``nn.ModuleDict``, and it can be accessed as such
print(layer.parametrizations)
print(layer.parametrizations.weight)

###############################################################################
# Each element of this ``nn.ModuleDict`` is a ``ParametrizationList``, which behaves like an
# ``nn.Sequential``. This list will allow us to concatenate parametrizations on one weight.
# Since this is a list, we can access the parametrizations indexing it. Here's
# where our ``Symmetric`` parametrization sits
print(layer.parametrizations.weight[0])

###############################################################################
# The other thing that we notice is that, if we print the parameters, we see that the
# parameter ``weight`` has been moved
print(dict(layer.named_parameters()))

###############################################################################
# It now sits under ``layer.parametrizations.weight.original``
print(layer.parametrizations.weight.original)

###############################################################################
# Besides these three small differences, the parametrization is doing exactly the same
# as our manual implementation
symmetric = Symmetric()
weight_orig = layer.parametrizations.weight.original
print(torch.dist(layer.weight, symmetric(weight_orig)))

###############################################################################
# Parametrizations are first-class citizens
# -----------------------------------------
#
# Since ``layer.parametrizations`` is an ``nn.ModuleList``, it means that the parametrizations
# are properly registered as submodules of the original module. As such, the same rules
# for registering parameters in a module apply to register a parametrization.
# For example, if a parametrization has parameters, these will be moved from CPU
# to CUDA when calling ``model = model.cuda()``.
#
# Caching the value of a parametrization
# --------------------------------------
#
# Parametrizations come with an inbuilt caching system via the context manager
# ``parametrize.cached()``
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
# Concatenating parametrizations
# ------------------------------
#
# Concatenating two parametrizations is as easy as registering them on the same tensor.
# We may use this to create more complex parametrizations from simpler ones. For example, the
# `Cayley map <https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map>`_
# maps the skew-symmetric matrices to the orthogonal matrices of positive determinant. We can
# concatenate ``Skew`` and a parametrization that implements the Cayley map to get a layer with
# orthogonal weights
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
print(torch.dist(X.T @ X, torch.eye(3)))  # X is orthogonal

###############################################################################
# This may also be used to prune a parametrized module, or to reuse parametrizations. For example,
# the matrix exponential maps the symmetric matrices to the Symmetric Positive Definite (SPD) matrices
# But the matrix exponential also maps the skew-symmetric matrices to the orthogonal matrices.
# Using these two facts, we may reuse the parametrizations before to our advantage
class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", MatrixExponential())
X = layer_orthogonal.weight
print(torch.dist(X.T @ X, torch.eye(3)))         # X is orthogonal

layer_spd = nn.Linear(3, 3)
parametrize.register_parametrization(layer_spd, "weight", Symmetric())
parametrize.register_parametrization(layer_spd, "weight", MatrixExponential())
X = layer_spd.weight
print(torch.dist(X, X.T))                        # X is symmetric
print((torch.symeig(X).eigenvalues > 0.).all())  # X is positive definite

###############################################################################
# Intializing parametrizations
# ----------------------------
#
# Parametrizations come with a mechanism to initialize them. If we implement a method
# ``right_inverse`` with signature
#
# .. code-block:: python
#
#     def right_inverse(self, X: Tensor) -> Tensor
#
# it will be used when assigning to the parametrized tensor.
#
# Let's upgrade our implementation of the ``Skew`` class to support this
class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)

    def right_inverse(self, A):
        # We assume that A is skew-symmetric
        # We take the upper-triangular elements, as these are those used in the forward
        return A.triu(1)

###############################################################################
# We may now initialize a layer that is parametrized with ``Skew``
layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
X = torch.rand(3, 3)
X = X - X.T                             # X is now skew-symmetric
layer.weight = X                        # Initialize layer.weight to be X
print(torch.dist(layer.weight, X))      # layer.weight == X

###############################################################################
# This ``right_inverse`` works as expected when we concatenate parametrizations.
# To see this, let's upgrade the Cayley parametrization to also support being initialized
class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):
        # Assume X skew-symmetric
        # (I + X)(I - X)^{-1}
        return torch.solve(self.Id + X, self.Id - X).solution

    def right_inverse(self, A):
        # Assume A orthogonal
        # See https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
        # (X - I)(X + I)^{-1}
        return torch.solve(X - self.Id, self.Id + X).solution

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", CayleyMap(3))
# Sample an orthogonal matrix with positive determinant
X = torch.empty(3, 3)
nn.init.orthogonal_(X)
if X.det() < 0.:
    X[0].neg_()
layer_orthogonal.weight = X
print(torch.dist(layer_orthogonal.weight, X))  # layer_orthogonal.weight == X

###############################################################################
# This initialization step can be written more succinctly as
layer_orthogonal.weight = nn.init.orthogonal_(layer_orthogonal.weight)

###############################################################################
# The name of this method comes from the fact that we would often expect
# that ``forward(right_inverse(X)) == X``. This is a direct way of rewriting that
# the forward afer the initalization with value ``X`` should return the value ``X``.
# This constraint is not strongly enforced in practice. In fact, at times, it might be of
# interest to relax this relation. For example, consider the following implementation
# of a randomized pruning method:
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
# In this case, it is not true that for every matrix A ``forward(right_inverse(A)) == A``.
# This is only true when the matrix ``A`` has zeros in the same positions as the mask.
# Even then, if we assign a tensor to a pruned parameter, it will comes as no surprise
# that tensor will be, in fact, pruned
layer = nn.Linear(3, 4)
X = torch.rand_like(layer.weight)
print(f"Initialization matrix:\n{X}")
parametrize.register_parametrization(layer, "weight", PruningParametrization(layer.weight))
layer.weight = X
print(f"\nInitialized weight:\n{layer.weight}")

###############################################################################
# Removing parametrizations
# -------------------------
#
# We may remove all the parametrizations from a parameter or a buffer in a module
# by using ``parametrize.remove_parametrizations()``
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
# When removing a parametrization, we may choose to leave the original parameter (i.e. that in
# ``layer.parametriations.weight.original``) rather than its parametrized version by setting
# the flag ``leave_parametrized=False``
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
