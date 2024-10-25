PyTorch를 활용한 자연어 처리를 위한 딥러닝
----------------------------------
**번역**: `임성연 <http://github.com/Choigapju>`_

이 튜토리얼 시리즈는 PyTorch를 활용한 딥러닝 프로그래밍의 핵심 개념들을 단계별로 안내합니다. 
여기서 다루는 많은 개념들(예를 들어, 계산 그래프 추상화와 자동 미분)은 PyTorch에만 국한된 것이 아니라 현존하는 모든 딥러닝 도구에 공통적으로 적용되는 원리입니다.

이 튜토리얼은 특히 딥러닝 프레임워크(예: TensorFlow, Theano, Keras, DyNet 등)로 코드를 작성해 본 경험이 전혀 없는 분들을 위한 자연어 처리에 초점을 맞추고 있습니다. 품사 태깅, 언어 모델링 등 핵심 자연어 처리 문제에 대한 기본적인 이해를 전제로 합니다. 또한 입문 수준의 인공지능 강좌(예: Russell과 Norvig의 교재에서 다루는 수준)에서 학습하는 정도의 신경망 지식을 갖추고 있다고 가정합니다.
일반적으로 이런 강좌들은 순전파 신경망의 기본적인 역전파 알고리즘을 다루며, 신경망이 선형 변환과 비선형 활성화 함수의 연쇄 구성이라는 점을 강조합니다. 본 튜토리얼의 주된 목표는 이러한 선수 지식을 바탕으로 여러분이 실제로 딥러닝 코드를 작성하기 시작할 수 있도록 안내하는 것입니다.

이 튜토리얼 시리즈는 특히, 어떤 딥러닝 프레임워크(예: TensorFlow, Theano, Keras, DyNet)로 코드를 작성해 본 경험이 전혀 없는 사람들을 위한 자연어 처리(NLP)에 초점을 맞추고 있습니다. 
이 튜토리얼은 품사 태깅, 언어 모델링 등 핵심 NLP 문제에 대한 실용적 지식을 전제로 합니다. 또한 입문 수준의 인공지능 강의(예: Russell과 Norvig의 교재에서 다루는 수준)에서 학습하는 정도의 신경망에 대한 이해를 가정합니다. 일반적으로 이러한 강의들은 순전파 신경망의 기본적인 역전파 알고리즘을 다루며, 신경망이 선형성과 비선형성의 연쇄적 구성이라는 점을 강조합니다. 이 튜토리얼의 목표는 이러한 선수 지식을 바탕으로 여러분이 실제로 딥러닝 코드를 작성하기 시작할 수 있도록 안내하는 것입니다.

참고드릴 점은 이 튜토리얼은 데이터가 아닌 *모델*에 관한 것입니다. 모든 모델에 대해, 
학습 과정에서 가중치가 어떻게 변화하는지 확인할 수 있도록 작은 차원의 테스트 예제들을 몇 가지 생성했습니다. 실제 데이터로 시도해보고 싶으시다면, 이 노트북의 모든 모델을 그대로 가져가서 여러분의 데이터에 적용해보실 수 있습니다.

1. pytorch_tutorial.py
	파이토치 입문
	https://tutorials.pytorch.kr/beginner/nlp/pytorch_tutorial.html

2. deep_learning_tutorial.py
	PyTorch를 활용한 딥러닝
	https://tutorials.pytorch.kr/beginner/nlp/deep_learning_tutorial.html

3. word_embeddings_tutorial.py
	단어 임베딩: 어휘의 의미 표현
	https://tutorials.pytorch.kr/beginner/nlp/word_embeddings_tutorial.html

4. sequence_models_tutorial.py
	순차 모델과 LSTM(장단기 메모리) 네트워크
	https://tutorials.pytorch.kr/beginner/nlp/sequence_models_tutorial.html

5. advanced_tutorial.py
	고급: 동적 의사결정과 양방향 LSTM CRF
	https://tutorials.pytorch.kr/beginner/nlp/advanced_tutorial.html