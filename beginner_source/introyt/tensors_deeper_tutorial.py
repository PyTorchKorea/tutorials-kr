"""
`Introduction <introyt1_tutorial.html>`_ ||
**Tensors** ||
`Autograd <autogradyt_tutorial.html>`_ ||
`Building Models <modelsyt_tutorial.html>`_ ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

Pytorch Tensor 소개
===============================
번역:
`이상윤 <https://github.com/falconlee236>`_

아래 영상이나 `youtube <https://www.youtube.com/watch?v=r7QDUPb2dCM>`__ 를 참고하세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/r7QDUPb2dCM" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

텐서는 Pytorch에서 중요한 추상 데이터 자료형입니다. 이 interactive
notebook은 ``torch.Tensor`` 클래스에 대한 심층적인 소개를 제공합니다.

먼저 가장 중요한 것은 Pytorch 모듈을 import 하는 것입니다. 또한 몇가지
예제에 사용할 math 모듈도 import 합니다.

"""

import torch
import math


#########################################################################
# Tensor 생성하기
# ----------------
#
# tensor를 생성하는 가장 간단한 방법은 ``torch.empty()`` 를 호출하는 것입니다:
# 

x = torch.empty(3, 4)
print(type(x))
print(x)


##########################################################################
# 방금 무엇을 한 것인지 들여다봅시다:
# 
# -  ``torch`` 모듈에 있는 수많은 메소드 중 하나를 사용해서 tesnor를 만들었습니다.
# -  이 tensor는 3개의 행과 4개의 열을 가진 2차원 tensor입니다.
# -  객체가 반환한 type은 ``torch.Tensor`` 이며 이는 ``torch.FloatTensor`` 의 별칭입니다.
#    기본적으로 Pytorch tensor는 32-bit 부동 소숫점 표현 실수로 채워집니다.
#    (아래에서 더 많은 데이터 자료형을 소개합니다)
# -  생성한 tensor를 출력하면 아마 무작위 값을 볼 수 있을 것 입니다.
#    ``torch.empty()`` 는 tensor를 위한 메모리를 할당해 주지만 임의의 값으로 초기화하지는 않습니다
#    - 그렇기 때문에 할당 당시에 메모리가 가지고 있는 값을 보는 것입니다.
# 
# 간략하게 tensor와 tensor의 차원 수, 그리고 각 tensor의 용어에 대해 알아봅시다:
# 
# -  때로는 1차원 tensor를 보게 될것인데 이는 *vector* 라고 합니다.
# -  이와 마찬가지로 2차원 tensor는 주로 *matrix* 라고 합니다.
# -  2차원보다 큰 차원을 가진 것들은 일반적으로 그냥 tensor라고 합니다.
# 
# 코딩 하면서 주로 tensor를 몇가지 값으로 초기화하고 싶을수가 있습니다.
# 일반적인 경우로는 모두 0으로 초기화 하거나, 모두 1로 초기화 하거나,
# 혹운 모두 무작위 값으로 초기화 할때가 있는데,
# 이때 ``torch`` 모듈은 이 모든 경우를 위한 메소드를 제공합니다:
# 

zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)


#########################################################################
# 이 메소드들은 우리가 기대하던 것들을 모두 수행합니다 - 0으로 모두 채워진 tensor,
# 1로 모두 채워진 tensor 그리고 0과 1사이의 무작위 값으로 채워진 tensor를 얻었습니다.
# 
# 무작위 Tensor들과 Seed사용하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 무작위 tensor에 대해 말하자면, 바로 앞에서 호출하는 ``torch.manual_seed()`` 를 눈치채셨나요?
# 특히 연구 환경에서 연구 결과의 재현 가능성에 대한 확신을 얻고 싶을때,
# 모델의 학습 가중치와 같은 무작위 값을 가진 tensor로 초기화 하는 것은 흔하거나 종종 일어나는 일입니다.
# 직접 무작위 난수 생성기의 seed를 설정하는 것이 다음 방법입니다. 다음 코드를 보며 더 자세히 알아봅시다:
# 

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)


############################################################################
# ``random1`` 과 ``random3`` 그리고 ``random2`` 과 ``random4`` ,
# 이 각각 서로 동일한 결과가 나온다는 것을 볼 수 있습니다.
# 무작위 난수 생성기의 seed를 수동으로 설정하면 난수가 재 설정되어 대부분의 환경에서
# 무작위 숫자에 의존하는 동일한 계산이 이루어지고 동일한 결과를 제공합니다.
# 
# 보다 자세한 정보는 다음 문서를 참고하세요 `PyTorch documentation on
# reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`__.
# 
# Tensor의 형태
# ~~~~~~~~~~~~~
# 
# 두개 혹은 그 이상의 tensor에 대한 연산을 수행할 때, tensor들은 같은 형태를 필요로 합니다
# - 다시 말해서 차원의 개수가 같아야 하고, 각 차원마다 원소의 수가 같아야 합니다.
# 그러기 위해서는 ``torch.*_like()`` 함수를 사용합니다.
# 

x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)


#########################################################################
# 위쪽의 코드 셀에 있는 것들 중에 첫 번째는 tensor에 있는 ``.shape`` 속성을 사용했습니다.
# 이 속성은 tesnor의 각 차원 크기에 대한 리스트를 포함합니다
# - 이 경우에, ``x`` 는 형태가 2 x 2 x 3 인 3차원 tensor입니다.
# 
# 그 아래에는 ``.empty_like()``, ``.zeros_like()``,
# ``.ones_like()``, and ``.rand_like()`` 메소드를 호출 합니다.
# ``.shape`` 속성을 통해서, 위의 메소드들이 동일한 차원값을 반환한다는 것을 검증할 수 있습니다.
# 
# 여기서 다루는 tensor를 생성하는 마지막 방법은 PyTorch collection
# 형식의 데이터를 직접적으로 명시하는 것 입니다:
# 

some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)


######################################################################
# ``torch.tensor()`` 는 이미 Python tuple이나 list 형태로 이루어진 데이터를
# 가지고 있는 경우 tensor를 만들기 가장 쉬운 방법입니다.
# 위에서 본것 처럼 중접된 형태의 collection 자료형은 다차원 tensor가 결과로 나옵니다.
# 
# .. note::
#      ``torch.tensor()`` 는 데이터의 복사본을 생성합니다.
# 
# Tensor 자료형
# ~~~~~~~~~~~~~~~~~
# 
# tensor의 자료형을 설정하는 것은 다양한 방식이 가능합니다.
# 

a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)


##########################################################################
# tensor의 자료형을 설정하는 가장 단순한 방식은 생성할 때 선택적 인자를 사용하는 것 입니다.
# 위에있는 cell의 첫번째 줄을 보면, tensor ``a`` 를
# ``dtype=torch.int16`` 자료형으로 설정했습니다. ``a`` 를 출력할 때,
# ``1.`` 대신에 ``1`` 로 가득찬 모습을 볼 수 있습니다
# - 파이썬에서 아랫 점이 없으면 실수 자료형이 아닌 정수 자료형을 의미합니다.
# 
# ``a`` 를 출력할때 또 한가지 주목할 점은,
# ``dtype`` 을 기본값 (32-bit 부동 소수점)
# 으로 남길 때와 다르게 tensor를 출력하는 경우
# 각 tensor의 ``dtype`` 을 명시한다는 것입니다.
# 
# tensor의 형태를 정수형 인자의 나열, 즉 이 인자를 tuple 자료형 형태로
# 묶는다는 것을 발견할 수 있습니다. 이것은 반드시 필요한 것은 아닙니다
# - Pytorch에서는 첫번째 인수로 tensor 형태라는 값을 의미하는 라벨이 없는 정수 인자를 여러개를 받습니다 -
# 하지만 선택 인수를 추가했을 때, 이 방식은 코드를 더 읽기 쉽게 만들 수 있습니다.
# 
# 자료형을 설정하는 다른 방밥은 ``.to()`` 메소드랑 함께 사용하는 것 입니다.
# 위쪽 셀에서 평범한 방식으로 무작위 실수 자료형 tensor ``b`` 를 생성합니다.
# 이어서 ``.to()`` 메소드를 사용해서 ``b`` 를 32-bit 정수 자료형으로 변환한 ``c`` 를 생성합니다.
# ``c`` 는 모든 ``b`` 의 값과 같은 값을 가지고 있지만 소숫점 아래 자리를 버린다는 점이 다릅니다.
# 
# 가능한 데이터 자료형은 다음을 포함합니다:
# 
# -  ``torch.bool``
# -  ``torch.int8``
# -  ``torch.uint8``
# -  ``torch.int16``
# -  ``torch.int32``
# -  ``torch.int64``
# -  ``torch.half``
# -  ``torch.float``
# -  ``torch.double``
# -  ``torch.bfloat``
# 
# Pytorch Tensor에서 산술 & 논리 연산
# ---------------------------------
# 
# 지금까지 tensor를 생성하는 몇가지 방식을 알아봤습니다…
# 이것을 가지고 무엇을 할 수 있을까요?
# 
# 먼저 기본적인 산술 연산을 알아보고,
# 그 다음 tensor가 단순 스칼라 값과 어떻게 상호작용 하는지 알아봅시다:
# 

ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)


##########################################################################
# 위에서 볼 수 있듯이 tensor들과 스칼라 값 사이 산술연산,
# 예를 들면 덧셈, 뺄셈, 곱셈, 나눗셈 그리고 거듭제곱은
# tensor의 각 원소에 나눠서 계산을 합니다.
# 이러한 연산의 결과는 tensor가 될 것이기 때문에,
# ``threes`` 변수를 생성하는 줄에서 처럼
# 일반적인 연산자 우선순위 규칙과 함께 연산자를 연결할 수 있습니다.
# 
# 두 tensor 사이 유사한 연산도 직관적으로 예상할 수 있는 방식으로 동작합니다:
# 

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)


##########################################################################
# 여기서 주목해야할 점은 이전 코드 cell에 있는 모든 tensor는 동일한 형태를 가져야 한다는 것 입니다.
# 만약 서로 다른 형태를 가진 tensor끼리 이진 연산을 수행한다면 무슨 일이 일어날까요?
# 
# .. note::
#      다음 cell은 run-time error가 발생합니다. 이것은 의도한 것입니다.
#
# ::
#
#    a = torch.rand(2, 3)
#    b = torch.rand(3, 2)
#
#    print(a * b)
#


##########################################################################
# 일반적인 경우에, 다른 형태의 tensor를 이러한 방식으로 연산할 수 없습니다.
# 심지어 위에 있는 cell에 있는 경우처럼 tensor가 서로 같은 개수의 원소를 가지고 있는 경우에도 연산할 수 없습니다.
# 
# 개요: Tensor Broadcasting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# .. note::
#      만약 NumPy의 ndarrays에서 사용하는 broadcasting 문법에 익숙하다면,
#      여기서도 같은 규칙이 적용된다는 것을 확인할 수 있습니다.
# 
# tensor는 같은 형태끼리만 연산이 가능하다는 규칙의 예외가 바로 *tensor broadcasting* 입니다.
# 다음은 그 예시입니다:
# 

rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)


#########################################################################
# 여기서 무슨 트릭이 사용되고 있는 것일까요?
# 어떻게 2x4 tensor에 1x4 tensor를 곱한 값을 얻을 수 있을까요?
# 
# Broadcasting은 서로 비슷한 형태를 가진 tensor사이 연산을 수행하는 방법입니다.
# 위의 예시를 보면, 행의 값이 1이고, 열의 값이 4인 tensor가
# 행의 값이 2이고, 열의 값이 4인 tensor의 *모든 행* 에 곱하게 됩니다.
# 
# 이것은 딥러닝에서 중요한 연산입니다.
# 일반적인 예시는 학습 가중치 tensor에 입력 tensor의 *배치* 를 곱하고,
# 배치의 각 인스턴스에 곱하기 연산을 개별적으로 적용한 이후
# 위의 (2, 4) \* (1, 4) tensor연산의 결과가 (2, 4)형태 tensor인 것처럼 -
# 동일한 형태의 학습 가중치 tesnor를 반환하는 것입니다.
# 
# Broadcasting의 규칙은 다음과 같습니다:
# 
# -  각 tensor는 최소한 1차원 이상을 반드시 가지고 있어야 합니다 - 빈 tensor는 사용할 수 없습니다.
# 
# -  두 tensor의 각 차원 크기 원소가 다음 조건을 만족하는지 확인하며 비교합니다. *이때 비교 순서는 맨 뒤에서부터 맨 앞으로 입니다;*
#
#    -  각 차원이 서로 동일합니다, *또는*
# 
#    -  각 차원중의 하나의 크기가 반드시 1입니다, *또는*
# 
#    -  tensor들 중 하나의 차원이 존재하지 않습니다.
# 
# 이전에 봤던 것처럼,
# 물론 동일한 형태를 가진 Tensor들은 자명하게 “broadcastable” 합니다.
# 
# 다음 예제는 위의 규칙을 준수하고
# broadcasting을 허용하는 몇가지 상황입니다.
# 

a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 세번째와 두번째 차원이 a랑 동일하고, 첫번째 차원은 존재하지 않습니다.
print(b)

c = a * torch.rand(   3, 1) # 세번째 차원 = 1이고, 두번째 차원은 a랑 동일합니다.
print(c)

d = a * torch.rand(   1, 2) # 세번째 차원이 a랑 동일하고, 두번째 차원 = 1입니다.
print(d)


#############################################################################
# 위의 예시에 있는 각 tensor의 값을 자세히 살펴봅시다:
#
# -  ``b`` 를 만드는 곱셈 연산은
#    ``a`` 의 모든 “layer” 에 broadcast 되었습니다.
# -  ``c`` 에 대해서, 연산은 ``a`` 의 모든 layer와 행에 대해서 broadcast 되었습니다
#    - 모든 열은 3개의 원소값 모두 동일합니다.
# -  ``d`` 에 대해서, 연산이 이전과 반대로 모든 layer와 열에 대해서 수행합니다
#    - 이재 모든 *행* 이 동일합니다.
# 
# broadcasting에 대한 더 많은 정보는, `PyTorch
# documentation <https://pytorch.org/docs/stable/notes/broadcasting.html>`__
# 에 있는 주제를 참고하세요.
# 
# 다음 예시는 broadcasting 시도가 실패한 사례입니다:
# 
# .. note::
#       다음 cell은 run-time error가 발생합니다. 이것은 의도한 것입니다.
#
# ::
#
#    a =     torch.ones(4, 3, 2)
#
#    b = a * torch.rand(4, 3)    # 차원은 반드시 맨 뒤 원소부터 맨 앞 원소로 차례대로 맞춰야 합니다.
#
#    c = a * torch.rand(   2, 3) # 세번째와 두번째 차원 모두 서로 다릅니다.
#
#    d = a * torch.rand((0, ))   # 비어있는 tensor는 broadcast 할 수 없습니다.
#


###########################################################################
# Tensor를 사용하는 다양한 연산들
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# PyTorch tensor는 tensor들 끼리 수행할 수 있는 300개 이상의
# 연산을 가지고 있습니다.
# 
# 다음 작은 예시는 주로 사용하는 연산 종류 몇개를 보여줍니다:
# 

# 공용 함수
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# 삼각 함수들 역 삼각 함수
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# 비트 연산
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# 비교 연산:
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # 많은 비교 연산자들은 broadcasting을 지원합니다!
print(torch.eq(d, e)) # bool 자료형을 가진 tensor를 반환합니다.

# 차원 감소 연산:
print('\nReduction ops:')
print(torch.max(d))        # 단일 원소 tensor를 반환합니다.
print(torch.max(d).item()) # 반환한 tensor로부터 값을 추출합니다.
print(torch.mean(d))       # 평균
print(torch.std(d))        # 표준 편차
print(torch.prod(d))       # 모든 숫자의 곱
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # 중복되지 않은 값들을 걸러냅니다.

# 벡터와 선형 대수 연산
v1 = torch.tensor([1., 0., 0.])         # x축 단위 벡터
v2 = torch.tensor([0., 1., 0.])         # y축 단위 벡터
m1 = torch.rand(2, 2)                   # 무작위 행렬
m2 = torch.tensor([[3., 0.], [0., 3.]]) # 단위 행렬에 3을 곱한 결과

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # z축 단위 벡터의 음수값 (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # m1 행렬을 3번 곱한 결과
print(torch.svd(m3))       # 특이값 분해


##################################################################################
# 이것은 수많은 연산의 일부분입니다.
# 더 자세한 내용이나 수학 함수의 전체적인 목록은, 다음
# `documentation <https://pytorch.org/docs/stable/torch.html#math-operations>`__
# 를 읽어주세요.
#
# Tensor의 값을 변경하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 대부분 tensor들의 이진 연산은 제3자의 새로운 tensor를 생성합니다.
# ``c = a * b`` ( ``a`` 와 ``b`` 는 tensor)연산을 수행할 때,
# 새로운 tensor ``c`` 는 다른 tensor와 구별되는 메모리 영역을 차지하게 됩니다.
# 
# 그럼에도 불구하고 tensor의 값을 변경하고 싶은 순간이 있을 수 있습니다 -
# 예를 들어, 중간 연산 결과 값을 버릴 수 있는 각 원소 단위 연산을 수행하는 경우가 있습니다.
# 이런 연산을 위해, 대부분의 수학 함수들은 tensor 내부의 값을
# 변경할 수 있는 함수 이름 맨 뒤에 밑줄 (``_``)이 추가된 버전을 가지고 있습니다.
# 
# 예시:
# 

a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # 이 연산은 메모리에 새로운 tensor를 생성합니다.
print(a)              # a는 변하지 않습니다.

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # 밑줄에 주목하세요.
print(b)              # b가 변합니다.


#######################################################################
# 산술 연산에서, 비슷한 행동을 하는 함수가 있습니다:
# 

a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)


##########################################################################
# 이러한 내부의 값을 변경하는 산술 함수는 다른 많은 함수들
# (e.g., ``torch.sin()``)처럼 ``torch`` 모듈의 메소드가 아니라
# ``torch.Tensor`` 객체의 메소드인 점에 주목해야 합니다.
# ``a.add_(b)`` 와 같은 경우처럼, *메소드를 호출하는 tensor는 값이 변경됩니다.*
# 
# 이미 존재하고 있는 메모리에 할당된 tensor에 계산 결과값을 저장하는 또 다른 옵션이 있습니다.
# tensor를 생성하는 메소드 뿐만 아니라 지금까지 이 문서에서 봤던 수많은 함수나 메소드는
# 결과 값을 받는 특정 tensor를 명시하는 ``out`` 이라는 인수를 가지고 있습니다.
# 만약 ``out`` tensor가 알맞는 형태와 ``dtype`` 을 가지고 있다면,
# 새로운 메모리 할당 없이 결과값이 저장됩니다:
# 

a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # c의 값이 변경되었습니다.

assert c is d           # c와 d가 서로 단순히 같은 값을 가지는지가 아니라 같은 객체인지 테스트합니다.
assert id(c) == old_id  # 새로운 c는 이전 객체와 확실히 같은 객체입니다.

torch.rand(2, 2, out=c) # 다시 한번 생성해봅시다!
print(c)                # c의 값이 다시 바뀌었습니다.
assert id(c) == old_id  # 하지만 여전히 같은 객체네요!


##########################################################################
# Tensor를 복사하기
# ---------------
# 
# 파이썬의 다른 객체와 마찬가지로 변수에 tensor를 할당하는 것은
# 변수가 tensor의 *label* 이 되고 값을 복사하지 않습니다. 다음 예시를 보시죠:
# 

a = torch.ones(2, 2)
b = a

a[0][1] = 561  # a의 값을 바꾸면...
print(b)       # ...b의 값이 바뀝니다.


######################################################################
# 하지만 만약 우리가 작업할 별도의 데이터 복사본을 원하면 어떻게 해야할까요?
# ``clone()`` 메소드가 당신이 찾던 해답이 될 것입니다:
# 

a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # 메모리 상의 다른 객체입니다...
print(torch.eq(a, b))  # ...하지만 여전히 같은 값을 가지고 있네요!

a[0][1] = 561          # a가 변경되었습니다...
print(b)               # ...하지만 여전히 b는 이전 값을 가지고 있네요.


#########################################################################
# **``clone()`` 메소드를 사용할 때 알아야할 중요한 점이 있습니다.**
# 만약 source tensor가 autograd를 가진다면 clone이 가능합니다.
# **이 부분은 autograd와 관련된 동영상에서 더 깊이 다룰 것 입니다.**
# 하지만 만약 자세한 내용을 간단히 알고 싶다면 계속 설명하겠습니다.
# 
# *대부분의 경우에서 이것이 바로 여러분이 원하는 것입니다.*
# 예를 들어, 만약 여러분의 모델이 그 모델의 ``forward()`` 메소드에 여러 갈래의 계산 경로가 있고
# 원본 tensor와 그것의 복제본 *모두* 가 모델의 결과 값에 기여를 한다면,
# 두 tensor에 대한 autograd를 설정하는 모델 학습을 활성화 합니다.
# 만약 여러분의 source tensor가 autograd를 사용할 수 있다면
# (일반적으로 학습 가중치의 집합이거나, 가중치를 포함하는 계산에서 파생된 경우),
# 여러분이 원하는 결과를 얻을 수 있습니다.
# 
# 반면에 원본 tensor나 그것의 복제본 *모두* 가 변화도를 추적할 필요가 없다면,
# source tensor의 autograd가 꺼져있다면
# clone을 사용할 수 있습니다.
# 
# 그러나 *세번째 경우* 가 있습니다:
# 기본적으로 변화도가 모든 것을 위해 켜져있지만 일부 지표를 생성하기 위해서
# 스트림 중간에서 일부 값을 생성하고 싶어 하는
# 여러분 모델의 ``forward()`` 함수에서 계산을 수행한다고 상상해 보세요.
# 이 경우에는 변화도를 추적하기 위해서 source tensor의 복제본을 원하지 *않을* 수 있습니다
# - 성능이 autograd의 히스토리 추적 기능을 끄면서 향상됩니다.
# 이 경우를 위해서는 source tensor에 ``.detach()`` 메소드를 사용할 수 있습니다:
# 

a = torch.rand(2, 2, requires_grad=True) # autograd를 켭니다.
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)


#########################################################################
# 여기서 무슨 일이 일어나는걸까요?
# 
# -  ``a`` 를  ``requires_grad=True`` 옵션을 킨 상태로 생성합니다.
#    **아직 이 선택적 인수를 다루지 않았지만, autograd 단원 동안만 다룰 것입니다.**
# -  When we print ``a``, it informs us that the property
#    ``requires_grad=True`` - this means that autograd and computation
#    history tracking are turned on.
# -  We clone ``a`` and label it ``b``. When we print ``b``, we can see
#    that it’s tracking its computation history - it has inherited
#    ``a``\ ’s autograd settings, and added to the computation history.
# -  We clone ``a`` into ``c``, but we call ``detach()`` first.
# -  Printing ``c``, we see no computation history, and no
#    ``requires_grad=True``.
# 
# The ``detach()`` method *detaches the tensor from its computation
# history.* It says, “do whatever comes next as if autograd was off.” It
# does this *without* changing ``a`` - you can see that when we print
# ``a`` again at the end, it retains its ``requires_grad=True`` property.
# 
# Moving to GPU
# -------------
# 
# One of the major advantages of PyTorch is its robust acceleration on
# CUDA-compatible Nvidia GPUs. (“CUDA” stands for *Compute Unified Device
# Architecture*, which is Nvidia’s platform for parallel computing.) So
# far, everything we’ve done has been on CPU. How do we move to the faster
# hardware?
# 
# First, we should check whether a GPU is available, with the
# ``is_available()`` method.
# 
# .. note::
#      If you do not have a CUDA-compatible GPU and CUDA drivers
#      installed, the executable cells in this section will not execute any
#      GPU-related code.
# 

if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')


##########################################################################
# Once we’ve determined that one or more GPUs is available, we need to put
# our data someplace where the GPU can see it. Your CPU does computation
# on data in your computer’s RAM. Your GPU has dedicated memory attached
# to it. Whenever you want to perform a computation on a device, you must
# move *all* the data needed for that computation to memory accessible by
# that device. (Colloquially, “moving the data to memory accessible by the
# GPU” is shorted to, “moving the data to the GPU”.)
# 
# There are multiple ways to get your data onto your target device. You
# may do it at creation time:
# 

if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda')
    print(gpu_rand)
else:
    print('Sorry, CPU only.')


##########################################################################
# By default, new tensors are created on the CPU, so we have to specify
# when we want to create our tensor on the GPU with the optional
# ``device`` argument. You can see when we print the new tensor, PyTorch
# informs us which device it’s on (if it’s not on CPU).
# 
# You can query the number of GPUs with ``torch.cuda.device_count()``. If
# you have more than one GPU, you can specify them by index:
# ``device='cuda:0'``, ``device='cuda:1'``, etc.
# 
# As a coding practice, specifying our devices everywhere with string
# constants is pretty fragile. In an ideal world, your code would perform
# robustly whether you’re on CPU or GPU hardware. You can do this by
# creating a device handle that can be passed to your tensors instead of a
# string:
# 

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)


#########################################################################
# If you have an existing tensor living on one device, you can move it to
# another with the ``to()`` method. The following line of code creates a
# tensor on CPU, and moves it to whichever device handle you acquired in
# the previous cell.
# 

y = torch.rand(2, 2)
y = y.to(my_device)


##########################################################################
# It is important to know that in order to do computation involving two or
# more tensors, *all of the tensors must be on the same device*. The
# following code will throw a runtime error, regardless of whether you
# have a GPU device available:
# 
# ::
# 
#    x = torch.rand(2, 2)
#    y = torch.rand(2, 2, device='gpu')
#    z = x + y  # exception will be thrown
# 


###########################################################################
# Manipulating Tensor Shapes
# --------------------------
# 
# Sometimes, you’ll need to change the shape of your tensor. Below, we’ll
# look at a few common cases, and how to handle them.
# 
# Changing the Number of Dimensions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# One case where you might need to change the number of dimensions is
# passing a single instance of input to your model. PyTorch models
# generally expect *batches* of input.
# 
# For example, imagine having a model that works on 3 x 226 x 226 images -
# a 226-pixel square with 3 color channels. When you load and transform
# it, you’ll get a tensor of shape ``(3, 226, 226)``. Your model, though,
# is expecting input of shape ``(N, 3, 226, 226)``, where ``N`` is the
# number of images in the batch. So how do you make a batch of one?
# 

a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)


##########################################################################
# The ``unsqueeze()`` method adds a dimension of extent 1.
# ``unsqueeze(0)`` adds it as a new zeroth dimension - now you have a
# batch of one!
# 
# So if that’s *un*\ squeezing? What do we mean by squeezing? We’re taking
# advantage of the fact that any dimension of extent 1 *does not* change
# the number of elements in the tensor.
# 

c = torch.rand(1, 1, 1, 1, 1)
print(c)


##########################################################################
# Continuing the example above, let’s say the model’s output is a
# 20-element vector for each input. You would then expect the output to
# have shape ``(N, 20)``, where ``N`` is the number of instances in the
# input batch. That means that for our single-input batch, we’ll get an
# output of shape ``(1, 20)``.
# 
# What if you want to do some *non-batched* computation with that output -
# something that’s just expecting a 20-element vector?
# 

a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)


#########################################################################
# You can see from the shapes that our 2-dimensional tensor is now
# 1-dimensional, and if you look closely at the output of the cell above
# you’ll see that printing ``a`` shows an “extra” set of square brackets
# ``[]`` due to having an extra dimension.
# 
# You may only ``squeeze()`` dimensions of extent 1. See above where we
# try to squeeze a dimension of size 2 in ``c``, and get back the same
# shape we started with. Calls to ``squeeze()`` and ``unsqueeze()`` can
# only act on dimensions of extent 1 because to do otherwise would change
# the number of elements in the tensor.
# 
# Another place you might use ``unsqueeze()`` is to ease broadcasting.
# Recall the example above where we had the following code:
# 
# ::
# 
#    a =     torch.ones(4, 3, 2)
# 
#    c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
#    print(c)
# 
# The net effect of that was to broadcast the operation over dimensions 0
# and 2, causing the random, 3 x 1 tensor to be multiplied element-wise by
# every 3-element column in ``a``.
# 
# What if the random vector had just been 3-element vector? We’d lose the
# ability to do the broadcast, because the final dimensions would not
# match up according to the broadcasting rules. ``unsqueeze()`` comes to
# the rescue:
# 

a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)       # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)             # broadcasting works again!


######################################################################
# The ``squeeze()`` and ``unsqueeze()`` methods also have in-place
# versions, ``squeeze_()`` and ``unsqueeze_()``:
# 

batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)


##########################################################################
# Sometimes you’ll want to change the shape of a tensor more radically,
# while still preserving the number of elements and their contents. One
# case where this happens is at the interface between a convolutional
# layer of a model and a linear layer of the model - this is common in
# image classification models. A convolution kernel will yield an output
# tensor of shape *features x width x height,* but the following linear
# layer expects a 1-dimensional input. ``reshape()`` will do this for you,
# provided that the dimensions you request yield the same number of
# elements as the input tensor has:
# 

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)


###############################################################################
# .. note::
#      The ``(6 * 20 * 20,)`` argument in the final line of the cell
#      above is because PyTorch expects a **tuple** when specifying a
#      tensor shape - but when the shape is the first argument of a method, it
#      lets us cheat and just use a series of integers. Here, we had to add the
#      parentheses and comma to convince the method that this is really a
#      one-element tuple.
# 
# When it can, ``reshape()`` will return a *view* on the tensor to be
# changed - that is, a separate tensor object looking at the same
# underlying region of memory. *This is important:* That means any change
# made to the source tensor will be reflected in the view on that tensor,
# unless you ``clone()`` it.
# 
# There *are* conditions, beyond the scope of this introduction, where
# ``reshape()`` has to return a tensor carrying a copy of the data. For
# more information, see the
# `docs <https://pytorch.org/docs/stable/torch.html#torch.reshape>`__.
# 


#######################################################################
# NumPy Bridge
# ------------
# 
# In the section above on broadcasting, it was mentioned that PyTorch’s
# broadcast semantics are compatible with NumPy’s - but the kinship
# between PyTorch and NumPy goes even deeper than that.
# 
# If you have existing ML or scientific code with data stored in NumPy
# ndarrays, you may wish to express that same data as PyTorch tensors,
# whether to take advantage of PyTorch’s GPU acceleration, or its
# efficient abstractions for building ML models. It’s easy to switch
# between ndarrays and PyTorch tensors:
# 

import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)


##########################################################################
# PyTorch creates a tensor of the same shape and containing the same data
# as the NumPy array, going so far as to keep NumPy’s default 64-bit float
# data type.
# 
# The conversion can just as easily go the other way:
# 

pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)


##########################################################################
# It is important to know that these converted objects are using *the same
# underlying memory* as their source objects, meaning that changes to one
# are reflected in the other:
# 

numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
