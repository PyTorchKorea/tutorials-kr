"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
**Building Models** ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

파이토치로 모델 만들기
============================

아래 비디오를 따라 하거나 또는 `youtube <https://www.youtube.com/watch?v=OSqIP-mOWOI>`__에서 확인하세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/OSqIP-mOWOI" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

``torch.nn.Module`` and ``torch.nn.Parameter``
----------------------------------------------

In this video, we’ll be discussing some of the tools PyTorch makes
available for building deep learning networks.

Except for ``Parameter``, the classes we discuss in this video are all
subclasses of ``torch.nn.Module``. This is the PyTorch base class meant
to encapsulate behaviors specific to PyTorch Models and their
components.

One important behavior of ``torch.nn.Module`` is registering parameters.
If a particular ``Module`` subclass has learning weights, these weights
are expressed as instances of ``torch.nn.Parameter``. The ``Parameter``
class is a subclass of ``torch.Tensor``, with the special behavior that
when they are assigned as attributes of a ``Module``, they are added to
the list of that modules parameters. These parameters may be accessed
through the ``parameters()`` method on the ``Module`` class.

As a simple example, here’s a very simple model with two linear layers
and an activation function. We’ll create an instance of it and ask it to
report on its parameters:

"""

import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)


#########################################################################
# This shows the fundamental structure of a PyTorch model: there is an
# ``__init__()`` method that defines the layers and other components of a
# model, and a ``forward()`` method where the computation gets done. Note
# that we can print the model, or any of its submodules, to learn about
# its structure.
#
# Common Layer Types
# ------------------
#
# Linear Layers
# ~~~~~~~~~~~~~
#
# The most basic type of neural network layer is a *linear* or *fully
# connected* layer. This is a layer where every input influences every
# output of the layer to a degree specified by the layer’s weights. If a
# model has *m* inputs and *n* outputs, the weights will be an *m* x *n*
# matrix. For example:
#

lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)


#########################################################################
# If you do the matrix multiplication of ``x`` by the linear layer’s
# weights, and add the biases, you’ll find that you get the output vector
# ``y``.
#
# One other important feature to note: When we checked the weights of our
# layer with ``lin.weight``, it reported itself as a ``Parameter`` (which
# is a subclass of ``Tensor``), and let us know that it’s tracking
# gradients with autograd. This is a default behavior for ``Parameter``
# that differs from ``Tensor``.
#
# Linear layers are used widely in deep learning models. One of the most
# common places you’ll see them is in classifier models, which will
# usually have one or more linear layers at the end, where the last layer
# will have *n* outputs, where *n* is the number of classes the classifier
# addresses.
#
# Convolutional Layers
# ~~~~~~~~~~~~~~~~~~~~
#
# *Convolutional* layers are built to handle data with a high degree of
# spatial correlation. They are very commonly used in computer vision,
# where they detect close groupings of features which the compose into
# higher-level features. They pop up in other contexts too - for example,
# in NLP applications, where a word’s immediate context (that is, the
# other words nearby in the sequence) can affect the meaning of a
# sentence.
#
# We saw convolutional layers in action in LeNet5 in an earlier video:
#

import torch.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


##########################################################################
# Let’s break down what’s happening in the convolutional layers of this
# model. Starting with ``conv1``:
#
# -  LeNet5 is meant to take in a 1x32x32 black & white image. **The first
#    argument to a convolutional layer’s constructor is the number of
#    input channels.** Here, it is 1. If we were building this model to
#    look at 3-color channels, it would be 3.
# -  A convolutional layer is like a window that scans over the image,
#    looking for a pattern it recognizes. These patterns are called
#    *features,* and one of the parameters of a convolutional layer is the
#    number of features we would like it to learn. **This is the second
#    argument to the constructor is the number of output features.** Here,
#    we’re asking our layer to learn 6 features.
# -  Just above, I likened the convolutional layer to a window - but how
#    big is the window? **The third argument is the window or kernel
#    size.** Here, the “5” means we’ve chosen a 5x5 kernel. (If you want a
#    kernel with height different from width, you can specify a tuple for
#    this argument - e.g., ``(3, 5)`` to get a 3x5 convolution kernel.)
#
# The output of a convolutional layer is an *activation map* - a spatial
# representation of the presence of features in the input tensor.
# ``conv1`` will give us an output tensor of 6x28x28; 6 is the number of
# features, and 28 is the height and width of our map. (The 28 comes from
# the fact that when scanning a 5-pixel window over a 32-pixel row, there
# are only 28 valid positions.)
#
# We then pass the output of the convolution through a ReLU activation
# function (more on activation functions later), then through a max
# pooling layer. The max pooling layer takes features near each other in
# the activation map and groups them together. It does this by reducing
# the tensor, merging every 2x2 group of cells in the output into a single
# cell, and assigning that cell the maximum value of the 4 cells that went
# into it. This gives us a lower-resolution version of the activation map,
# with dimensions 6x14x14.
#
# Our next convolutional layer, ``conv2``, expects 6 input channels
# (corresponding to the 6 features sought by the first layer), has 16
# output channels, and a 3x3 kernel. It puts out a 16x12x12 activation
# map, which is again reduced by a max pooling layer to 16x6x6. Prior to
# passing this output to the linear layers, it is reshaped to a 16 \* 6 \*
# 6 = 576-element vector for consumption by the next layer.
#
# There are convolutional layers for addressing 1D, 2D, and 3D tensors.
# There are also many more optional arguments for a conv layer
# constructor, including stride length(e.g., only scanning every second or
# every third position) in the input, padding (so you can scan out to the
# edges of the input), and more. See the
# `documentation <https://pytorch.org/docs/stable/nn.html#convolution-layers>`__
# for more information.
#
# Recurrent Layers
# ~~~~~~~~~~~~~~~~
#
# *Recurrent neural networks* (or *RNNs)* are used for sequential data -
# anything from time-series measurements from a scientific instrument to
# natural language sentences to DNA nucleotides. An RNN does this by
# maintaining a *hidden state* that acts as a sort of memory for what it
# has seen in the sequence so far.
#
# The internal structure of an RNN layer - or its variants, the LSTM (long
# short-term memory) and GRU (gated recurrent unit) - is moderately
# complex and beyond the scope of this video, but we’ll show you what one
# looks like in action with an LSTM-based part-of-speech tagger (a type of
# classifier that tells you if a word is a noun, verb, etc.):
#

class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


########################################################################
# The constructor has four arguments:
#
# -  ``vocab_size`` is the number of words in the input vocabulary. Each
#    word is a one-hot vector (or unit vector) in a
#    ``vocab_size``-dimensional space.
# -  ``tagset_size`` is the number of tags in the output set.
# -  ``embedding_dim`` is the size of the *embedding* space for the
#    vocabulary. An embedding maps a vocabulary onto a low-dimensional
#    space, where words with similar meanings are close together in the
#    space.
# -  ``hidden_dim`` is the size of the LSTM’s memory.
#
# The input will be a sentence with the words represented as indices of
# one-hot vectors. The embedding layer will then map these down to an
# ``embedding_dim``-dimensional space. The LSTM takes this sequence of
# embeddings and iterates over it, fielding an output vector of length
# ``hidden_dim``. The final linear layer acts as a classifier; applying
# ``log_softmax()`` to the output of the final layer converts the output
# into a normalized set of estimated probabilities that a given word maps
# to a given tag.
#
# If you’d like to see this network in action, check out the `Sequence
# Models and LSTM
# Networks <https://tutorials.pytorch.kr/beginner/nlp/sequence_models_tutorial.html>`__
# tutorial on pytorch.org.
#
# Transformers
# ~~~~~~~~~~~~
#
# *Transformers* are multi-purpose networks that have taken over the state
# of the art in NLP with models like BERT. A discussion of transformer
# architecture is beyond the scope of this video, but PyTorch has a
# ``Transformer`` class that allows you to define the overall parameters
# of a transformer model - the number of attention heads, the number of
# encoder & decoder layers, dropout and activation functions, etc. (You
# can even build the BERT model from this single class, with the right
# parameters!) The ``torch.nn.Transformer`` class also has classes to
# encapsulate the individual components (``TransformerEncoder``,
# ``TransformerDecoder``) and subcomponents (``TransformerEncoderLayer``,
# ``TransformerDecoderLayer``). For details, check out the
# `documentation <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__
# on transformer classes, and the relevant
# `tutorial <https://tutorials.pytorch.kr/beginner/transformer_tutorial.html>`__
# on pytorch.org.
#
# Other Layers and Functions
# --------------------------
#
# Data Manipulation Layers
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# There are other layer types that perform important functions in models,
# but don’t participate in the learning process themselves.
#
# **Max pooling** (and its twin, min pooling) reduce a tensor by combining
# cells, and assigning the maximum value of the input cells to the output
# cell (we saw this). For example:
#

my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))


#########################################################################
# 위의 값을 자세히 보면, 맥스풀링된 출력의 각 값이 6x6 입력의 각 사분면에서 
# 최대값이라는 것을 알 수 있습니다.
#
# **정규화 레이어**는 한 레이어의 출력을 다른 레이어에 전달하기 전에 다시 중심화하고 
# 정규화합니다. 중간 텐서를 중심화하고 스케일링하는 것은 기울기 폭발/소실 없이 더 
# 높은 학습률을 사용할 수 있게 하는 등 여러 가지 유익한 효과를 제공합니다.
#

my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())



##########################################################################
# 위의 셀을 실행하면 입력 텐서에 큰 스케일링 요소와 오프셋을 추가했습니다.
# 입력 텐서의 ``mean()`` 값이 약 15에 가까운 것을 볼 수 있습니다.
# 이를 정규화 레이어를 통해 실행하면 값들이 더 작아지고 0 주위로 그룹화됩니다.
# 실제로 평균은 매우 작아야 합니다 (> 1e-8).
#
# 이는 유익한데, 왜냐하면 많은 활성화 함수들(아래에서 논의)은 0 근처에서
# 가장 강한 기울기를 갖지만, 때때로 입력이 0에서 멀리 떨어지게 하는 경우
# 기울기 소실 또는 폭발 문제가 발생할 수 있기 때문입니다.
# 데이터를 가장 가파른 기울기 주변에 유지하면 일반적으로 더 빠르고,
# 더 나은 학습과 더 높은 학습률이 가능합니다.
#
# **드롭아웃 레이어**는 모델 내에서 *희소 표현*을 장려하기 위한 도구입니다.
# 즉, 더 적은 데이터로 추론을 수행하도록 모델을 푸시하는 것입니다.
#
# 드롭아웃 레이어는 학습 중에 입력 텐서의 일부를 무작위로 설정하여 작동합니다
# - 드롭아웃 레이어는 항상 추론 시에는 꺼져 있습니다.
# 이는 모델이 이 마스킹되거나 축소된 데이터셋을 학습하도록 강제합니다.
# 예를 들어:
# 

my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))


##########################################################################
# 위에서 드롭아웃이 샘플 텐서에 미치는 효과를 볼 수 있습니다. 개별 가중치가 
# 드롭아웃될 확률을 설정하기 위해 선택적으로 `p` 인수를 사용할 수 있으며, 
# 설정하지 않으면 기본값은 0.5입니다.
#
# 활성화 함수
# ~~~~~~~~~~~~~~~~~~~~
#
# 활성화 함수는 딥러닝을 가능하게 만듭니다. 신경망은 사실 많은 파라미터를 
# 가진 *수학적 함수를 시뮬레이션*하는 프로그램입니다. 만약 우리가 텐서를 
# 레이어 가중치로 반복적으로 곱하기만 한다면, *선형 함수*만을 시뮬레이션할 
# 수 있을 뿐입니다. 게다가, 모든 레이어를 하나의 행렬 곱셈으로 축소할 수 
# 있기 때문에 여러 레이어를 가질 필요가 없을 것입니다. 레이어 사이에 
# *비선형* 활성화 함수를 삽입하는 것이 딥러닝 모델이 단순히 선형 함수가 
# 아닌 어떤 함수든 시뮬레이션할 수 있게 하는 요소입니다.
#
# `torch.nn.Module`은 ReLU 및 그 변형들, Tanh, Hardtanh, sigmoid 등의 
# 주요 활성화 함수를 캡슐화한 객체를 포함하고 있습니다. 또한, 모델의 출력 
# 단계에서 가장 유용한 Softmax와 같은 다른 함수들도 포함하고 있습니다.
#
# 손실 함수
# ~~~~~~~~~~~~~~
#
# 손실 함수는 모델의 예측이 정답과 얼마나 차이가 나는지를 알려줍니다. 
# PyTorch에는 일반적인 MSE (평균 제곱 오차 = L2 노름), 교차 엔트로피 
# 손실, 그리고 분류기에 유용한 음의 가능도 손실 등 다양한 손실 함수가 포함되어 있습니다.
#