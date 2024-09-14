PyTorch를 이용한 NLP를 위한 딥러닝
**********************************
**저자**: `Robert Guthrie <https://github.com/rguthrie3/DeepLearningForNLPInPytorch>`_
**번역**: `오수연 <github.com/oh5221>`_

이 튜토리얼은 PyTorch를 사용한 딥러닝 프로그램의 주요 아이디어에 대해
차근차근 살펴볼 것입니다. 많은 개념들(계산 그래프 추상화 및
autograd)은 PyTorch에서만 제공하는 것이 아니며, 이미 공개된
딥러닝 toolkit과 관련이 있습니다.  

이 튜토리얼은 딥러닝 프레임워크(예: Tensorflow, Theano, Keras,
Dynet)에서 어떤 코드도 작성해 본 적이 없는 사람들을 
위한 NLP에 특별히 초점을 맞추어 작성하였습니다. 튜토리얼을 위해 NLP의
핵심 문제에 대한 실무 지식이 필요합니다: 품사 태깅, 언어 모델링 등. 또한
AI 입문 수업 수준 (Russel과 Norvig 책에 나오는 것 같은) 신경망 친숙도가 필요합니다. 일반적으로,
feed-forward 신경망에 대한 기본적인 역전파 알고리즘을
다루고, 선형성과 비선형성의 연쇄적인 구성이라는 점을
강조합니다. 이 튜토리얼은 이런 필수적인 지식이 있는 상태에서
딥러닝 코드 작성을 시작하는 것을 목표로 합니다.

이 튜토리얼이 데이터가 아니라 *모델* 에 관한 것임에 주의해야 합니다. 모든
모델에 있어, 단지 작은 차원을 가진 몇 가지 예제만을 만들어 훈련 시
가중치 변화를 볼 수 있게 합니다. 만약 실제 데이터를 갖고 있다면,
이 노트북의 모델 중 하나를 뽑아
사용해 볼 수 있을 것입니다.


.. toctree::
    :hidden:

    /beginner/nlp/pytorch_tutorial
    /beginner/nlp/deep_learning_tutorial
    /beginner/nlp/word_embeddings_tutorial
    /beginner/nlp/sequence_models_tutorial
    /beginner/nlp/advanced_tutorial


.. galleryitem:: /beginner/nlp/pytorch_tutorial.py
    :intro: 모든 딥러닝은 행렬의 일반화인 Tensor에 대한 계산입니다.

.. galleryitem:: /beginner/nlp/deep_learning_tutorial.py
    :intro: 딥러닝은 선형성과 비선형성을 영리하게 조합하는 것으로 구성됩니다. 비선형성 도입의 소개

.. galleryitem:: /beginner/nlp/word_embeddings_tutorial.py
    :intro: 단어 임베딩은 실수의 dense vector로, vocabulary(단어 집합)의 단어 당 하나씩입니다. NLP에서는 거의 feature 대부분의 경우에 해당합니다.

.. galleryitem:: /beginner/nlp/sequence_models_tutorial.py
    :intro: 이 시점에서, 다양한 feed-forward 네트워크를 보았습니다. 즉, 네트워크에 의해 유지되는 상태가 없습니다.


.. galleryitem:: /beginner/nlp/advanced_tutorial.py
    :intro: 동적 vs. 정적 딥러닝 Toolkits. PyTorch는 *동적* 신경망 키트입니다.


.. raw:: html

    <div style='clear:both'></div>
