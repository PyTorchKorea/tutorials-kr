PyTorch를 이용한 NLP 딥러닝
-----------------------

이 튜토리얼들은 PyTorch를 사용한 딥러닝 프로그래밍의 핵심 아이디어들을 안내합니다.
계산 그래프 추상화나 자동 미분(autograd)과 같은 개념들은 파이토치에만 국한된 것이 아니라, 대부분의 딥러닝 프레임워크에서 공통적으로 찾아볼 수 있는 핵심 요소들입니다.

이 튜토리얼들은 특히 딥러닝 프레임워크(예: TensorFlow, Theano, Keras, DyNet 등)로 코딩을 해본 경험이 전혀 없는 사람들을 위한 자연어 처리(NLP)에 초점을 맞추고 있습니다. 이 튜토리얼들은 품사 태깅(part-of-speech tagging), 언어 모델링(language modeling) 등과 같은 NLP의 핵심 문제들에 대한 기본적인 이해를 전제로 합니다. 또한, 이 튜토리얼들은 인공지능 입문 수준(예: Russell과 Norvig의 교재 수준)의 신경망에 대한 기본 지식을 갖추고 있음을 가정하고 있습니다. 일반적으로, 이러한 코스들은 순방향 신경망(feed-forward neural networks)에서의 기본적인 역전파(backpropagation) 알고리즘을 다루며, 신경망이 선형성과 비선형성의 연쇄적 구성으로 이루어져 있다는 점을 강조합니다.
이 튜토리얼은 앞서 언급한 선수 지식을 갖추고 있다는 전제 하에, 여러분이 딥러닝 코드를 작성하기 시작할 수 있도록 돕는 것을 목표로 합니다.

이 튜토리얼들은 데이터가 아닌, *모델*에 관해 작성되었음을 알립니다. 모든 모델에 대해, 작은 차원의 몇 가지 테스트 예제들이 제공되어 학습 과정에서 가중치가 어떻게 변화하는지 확인할 수 있습니다. 만약 여러분이 실제로 사용해 보고 싶은 데이터가 있다면, 이 노트북에서 제시된 모델들을 쉽게 추출하여 해당 데이터에 적용할 수 있을 것입니다.

1. pytorch_tutorial.py
   PyTorch 소개
   https://tutorials.pytorch.kr/beginner/nlp/pytorch_tutorial.html

2. deep_learning_tutorial.py
   PyTorch를 이용한 딥러닝
   https://tutorials.pytorch.kr/beginner/nlp/deep_learning_tutorial.html

3. word_embeddings_tutorial.py
   단어 임베딩: 어휘의 의미 인코딩
   https://tutorials.pytorch.kr/beginner/nlp/word_embeddings_tutorial.html

4. sequence_models_tutorial.py
   순차 모델과 LSTM 네트워크
   https://tutorials.pytorch.kr/beginner/nlp/sequence_models_tutorial.html

5. advanced_tutorial.py
   고급 과정: 동적 결정 및 Bi-LSTM CRF
   https://tutorials.pytorch.kr/beginner/nlp/advanced_tutorial.html