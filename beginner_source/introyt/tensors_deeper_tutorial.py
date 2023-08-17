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
# The simplest way to set the underlying data type of a tensor is with an
# optional argument at creation time. In the first line of the cell above,
# we set ``dtype=torch.int16`` for the tensor ``a``. When we print ``a``,
# we can see that it’s full of ``1`` rather than ``1.`` - Python’s subtle
# cue that this is an integer type rather than floating point.
# 
# Another thing to notice about printing ``a`` is that, unlike when we
# left ``dtype`` as the default (32-bit floating point), printing the
# tensor also specifies its ``dtype``.
# 
# You may have also spotted that we went from specifying the tensor’s
# shape as a series of integer arguments, to grouping those arguments in a
# tuple. This is not strictly necessary - PyTorch will take a series of
# initial, unlabeled integer arguments as a tensor shape - but when adding
# the optional arguments, it can make your intent more readable.
# 
# The other way to set the datatype is with the ``.to()`` method. In the
# cell above, we create a random floating point tensor ``b`` in the usual
# way. Following that, we create ``c`` by converting ``b`` to a 32-bit
# integer with the ``.to()`` method. Note that ``c`` contains all the same
# values as ``b``, but truncated to integers.
# 
# Available data types include:
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
# Math & Logic with PyTorch Tensors
# ---------------------------------
# 
# Now that you know some of the ways to create a tensor… what can you do
# with them?
# 
# Let’s look at basic arithmetic first, and how tensors interact with
# simple scalars:
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
# As you can see above, arithmetic operations between tensors and scalars,
# such as addition, subtraction, multiplication, division, and
# exponentiation are distributed over every element of the tensor. Because
# the output of such an operation will be a tensor, you can chain them
# together with the usual operator precedence rules, as in the line where
# we create ``threes``.
# 
# Similar operations between two tensors also behave like you’d
# intuitively expect:
# 

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)


##########################################################################
# It’s important to note here that all of the tensors in the previous code
# cell were of identical shape. What happens when we try to perform a
# binary operation on tensors if dissimilar shape?
# 
# .. note::
#      The following cell throws a run-time error. This is intentional.
#
# ::
#
#    a = torch.rand(2, 3)
#    b = torch.rand(3, 2)
#
#    print(a * b)
#


##########################################################################
# In the general case, you cannot operate on tensors of different shape
# this way, even in a case like the cell above, where the tensors have an
# identical number of elements.
# 
# In Brief: Tensor Broadcasting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# .. note::
#      If you are familiar with broadcasting semantics in NumPy
#      ndarrays, you’ll find the same rules apply here.
# 
# The exception to the same-shapes rule is *tensor broadcasting.* Here’s
# an example:
# 

rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)


#########################################################################
# What’s the trick here? How is it we got to multiply a 2x4 tensor by a
# 1x4 tensor?
# 
# Broadcasting is a way to perform an operation between tensors that have
# similarities in their shapes. In the example above, the one-row,
# four-column tensor is multiplied by *both rows* of the two-row,
# four-column tensor.
# 
# This is an important operation in Deep Learning. The common example is
# multiplying a tensor of learning weights by a *batch* of input tensors,
# applying the operation to each instance in the batch separately, and
# returning a tensor of identical shape - just like our (2, 4) \* (1, 4)
# example above returned a tensor of shape (2, 4).
# 
# The rules for broadcasting are:
# 
# -  Each tensor must have at least one dimension - no empty tensors.
# 
# -  Comparing the dimension sizes of the two tensors, *going from last to
#    first:*
# 
#    -  Each dimension must be equal, *or*
# 
#    -  One of the dimensions must be of size 1, *or*
# 
#    -  The dimension does not exist in one of the tensors
# 
# Tensors of identical shape, of course, are trivially “broadcastable”, as
# you saw earlier.
# 
# Here are some examples of situations that honor the above rules and
# allow broadcasting:
# 

a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(d)


#############################################################################
# Look closely at the values of each tensor above: 
#
# -  The multiplication operation that created ``b`` was 
#    broadcast over every “layer” of ``a``.
# -  For ``c``, the operation was broadcast over ever layer and row of
#    ``a`` - every 3-element column is identical. 
# -  For ``d``, we switched it around - now every *row* is identical,
#    across layers and columns.
# 
# For more information on broadcasting, see the `PyTorch
# documentation <https://pytorch.org/docs/stable/notes/broadcasting.html>`__
# on the topic.
# 
# Here are some examples of attempts at broadcasting that will fail:
# 
# .. note::
#       The following cell throws a run-time error. This is intentional.
#
# ::
#
#    a =     torch.ones(4, 3, 2)
#
#    b = a * torch.rand(4, 3)    # dimensions must match last-to-first
#
#    c = a * torch.rand(   2, 3) # both 3rd & 2nd dims different
#
#    d = a * torch.rand((0, ))   # can't broadcast with an empty tensor
#


###########################################################################
# More Math with Tensors
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# PyTorch tensors have over three hundred operations that can be performed
# on them.
# 
# Here is a small sample from some of the major categories of operations:
# 

# common functions
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# trigonometric functions and their inverses
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# bitwise operations
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# comparisons:
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # many comparison ops support broadcasting!
print(torch.eq(d, e)) # returns a tensor of type bool

# reductions:
print('\nReduction ops:')
print(torch.max(d))        # returns a single-element tensor
print(torch.max(d).item()) # extracts the value from the returned tensor
print(torch.mean(d))       # average
print(torch.std(d))        # standard deviation
print(torch.prod(d))       # product of all numbers
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # filter unique elements

# vector and linear algebra operations
v1 = torch.tensor([1., 0., 0.])         # x unit vector
v2 = torch.tensor([0., 1., 0.])         # y unit vector
m1 = torch.rand(2, 2)                   # random matrix
m2 = torch.tensor([[3., 0.], [0., 3.]]) # three times identity matrix

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # negative of z unit vector (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # 3 times m1
print(torch.svd(m3))       # singular value decomposition


##################################################################################
# This is a small sample of operations. For more details and the full inventory of
# math functions, have a look at the
# `documentation <https://pytorch.org/docs/stable/torch.html#math-operations>`__.
# 
# Altering Tensors in Place
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Most binary operations on tensors will return a third, new tensor. When
# we say ``c = a * b`` (where ``a`` and ``b`` are tensors), the new tensor
# ``c`` will occupy a region of memory distinct from the other tensors.
# 
# There are times, though, that you may wish to alter a tensor in place -
# for example, if you’re doing an element-wise computation where you can
# discard intermediate values. For this, most of the math functions have a
# version with an appended underscore (``_``) that will alter a tensor in
# place.
# 
# For example:
# 

a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # this operation creates a new tensor in memory
print(a)              # a has not changed

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # note the underscore
print(b)              # b has changed


#######################################################################
# For arithmetic operations, there are functions that behave similarly:
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
# Note that these in-place arithmetic functions are methods on the
# ``torch.Tensor`` object, not attached to the ``torch`` module like many
# other functions (e.g., ``torch.sin()``). As you can see from
# ``a.add_(b)``, *the calling tensor is the one that gets changed in
# place.*
# 
# There is another option for placing the result of a computation in an
# existing, allocated tensor. Many of the methods and functions we’ve seen
# so far - including creation methods! - have an ``out`` argument that
# lets you specify a tensor to receive the output. If the ``out`` tensor
# is the correct shape and ``dtype``, this can happen without a new memory
# allocation:
# 

a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # contents of c have changed

assert c is d           # test c & d are same object, not just containing equal values
assert id(c) == old_id  # make sure that our new c is the same object as the old one

torch.rand(2, 2, out=c) # works for creation too!
print(c)                # c has changed again
assert id(c) == old_id  # still the same object!


##########################################################################
# Copying Tensors
# ---------------
# 
# As with any object in Python, assigning a tensor to a variable makes the
# variable a *label* of the tensor, and does not copy it. For example:
# 

a = torch.ones(2, 2)
b = a

a[0][1] = 561  # we change a...
print(b)       # ...and b is also altered


######################################################################
# But what if you want a separate copy of the data to work on? The
# ``clone()`` method is there for you:
# 

a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561          # a changes...
print(b)               # ...but b is still all ones


#########################################################################
# **There is an important thing to be aware of when using ``clone()``.**
# If your source tensor has autograd, enabled then so will the clone.
# **This will be covered more deeply in the video on autograd,** but if
# you want the light version of the details, continue on.
# 
# *In many cases, this will be what you want.* For example, if your model
# has multiple computation paths in its ``forward()`` method, and *both*
# the original tensor and its clone contribute to the model’s output, then
# to enable model learning you want autograd turned on for both tensors.
# If your source tensor has autograd enabled (which it generally will if
# it’s a set of learning weights or derived from a computation involving
# the weights), then you’ll get the result you want.
# 
# On the other hand, if you’re doing a computation where *neither* the
# original tensor nor its clone need to track gradients, then as long as
# the source tensor has autograd turned off, you’re good to go.
# 
# *There is a third case,* though: Imagine you’re performing a computation
# in your model’s ``forward()`` function, where gradients are turned on
# for everything by default, but you want to pull out some values
# mid-stream to generate some metrics. In this case, you *don’t* want the
# cloned copy of your source tensor to track gradients - performance is
# improved with autograd’s history tracking turned off. For this, you can
# use the ``.detach()`` method on the source tensor:
# 

a = torch.rand(2, 2, requires_grad=True) # turn on autograd
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)


#########################################################################
# What’s happening here?
# 
# -  We create ``a`` with ``requires_grad=True`` turned on. **We haven’t
#    covered this optional argument yet, but will during the unit on
#    autograd.**
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
