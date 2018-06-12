Deep Learning for NLP with Pytorch
**********************************
**저자**: `Robert Guthrie <https://github.com/rguthrie3/DeepLearningForNLPInPytorch>`_


 이 튜토리얼은 당신을 PyTorch를 이용한 딥러닝 프로그래밍의 핵심 개념으로 들어가게 해줄 것입니다. 
 (computation graph abstraction(연산 그래프 추상화)와 autograd와 같은) 많은 개념들은 
 PyTorch에서는 독특하지 않고 외부의 어떠한 딥러닝 툴키트와도 연관 있습니다. 

저는 이 튜토리얼을 어떠한 딥 러닝 프레임워크(예를 들어, TensorFlow, Theano, Keras, Dynet)에서도 
코드를 작성해본 적이 없는 사람들이 특히 NLP: 문장을 형태소나 품사별로 나누기(part-of-speech tagging), 
언어 모델링 등에 집중할 수 있도록 작성했습니다. 이것은 핵심 NLP 문제들의 실용지식을 추정합니다. 
이것은 또한 (Russel이나 Norvig book의 것처럼) 도입 AI 클래스의 수준에서 신경망에 익숙하다고 가정합니다. 
대부분, 이 과정들은 피드 포워드(feed-forward) 신경망의 기초 역전파 알고리즘을 다루고 
그들이 선형성과 비선형성으로 이루어진 사슬이라는 것을 강조합니다. 이 튜토리얼의 목적은 이러한 필수 지식을 
갖추고 있는 경우, 당신이 딥 러닝 코드를 작성하는 것을 시작할 수 있도록 하는 것입니다.

이것이 데이터가 아니라 모델이라는 것에 주목해 주세요. 모든 모델을 위해서, 
저는 당신이 그것이 학습되면서 가중치가 어떻게 바뀌는지를 볼 수 있도록 
작은 차원성을 이용해 몇 개의 실험 예를 만들었을 뿐입니다. 만약 당신이 시도해보고 싶은 
어떤 실제 데이터를 가지고 있다면, 당신은 이 노트북에서 
어떠한 모델을 아무렇지 않게 꺼내서 사용할 수 있어야 합니다.


.. toctree::
    :hidden:

    /beginner/nlp/pytorch_tutorial
    /beginner/nlp/deep_learning_tutorial
    /beginner/nlp/word_embeddings_tutorial
    /beginner/nlp/sequence_models_tutorial
    /beginner/nlp/advanced_tutorial


.. galleryitem:: /beginner/nlp/pytorch_tutorial.py
    :intro: All of deep learning is computations on tensors, which are generalizations of a matrix that can be

.. galleryitem:: /beginner/nlp/deep_learning_tutorial.py
    :intro: Deep learning consists of composing linearities with non-linearities in clever ways. The introduction of non-linearities allows

.. galleryitem:: /beginner/nlp/word_embeddings_tutorial.py
    :intro: Word embeddings are dense vectors of real numbers, one per word in your vocabulary. In NLP, it is almost always the case that your features are

.. galleryitem:: /beginner/nlp/sequence_models_tutorial.py
    :intro: At this point, we have seen various feed-forward networks. That is, there is no state maintained by the network at all.

.. galleryitem:: /beginner/nlp/advanced_tutorial.py
    :intro: Dynamic versus Static Deep Learning Toolkits. Pytorch is a *dynamic* neural network kit.


.. raw:: html

    <div style='clear:both'></div>
