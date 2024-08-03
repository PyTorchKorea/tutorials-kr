PyTorch를 이용한 NLP 딥러닝
-----------------------

이 튜토리얼들은 PyTorch를 사용한 딥러닝 프로그래밍의 핵심 아이디어들을 안내합니다.
계산 그래프 추상화나 자동 미분(autograd)과 같은 개념들은 파이토치에만 국한된 것이 아니라, 대부분의 딥러닝 프레임워크에서 공통적으로 찾아볼 수 있는 핵심 요소들입니다.

이 튜토리얼들은 특히 딥러닝 프레임워크(예: TensorFlow, Theano, Keras, DtNet 등)로 코딩을 해본 경험이 전혀 없는 사람들을 위한 자연어 처리(NLP)에 초점을 맞추고 있습니다. 이 튜토리얼들은 품사 태깅(part-of-speech tagging), 언어 모델링(language modeling) 등과 같은 NLP의 핵심 문제들에 대한 기본적인 이해를 전제로 합니다. 또한, 이 튜토리얼들은 인공지능 입문 수준(예: Russell, Norvig의 교재 수준)의 신경망에 대한 기본 지식을 갖추고 있음을 가정하고 있습니다.

Deep Learning for NLP with Pytorch
----------------------------------

These tutorials will walk you through the key ideas of deep learning
programming using Pytorch. Many of the concepts (such as the computation
graph abstraction and autograd) are not unique to Pytorch and are
relevant to any deep learning toolkit out there.

They are focused specifically on NLP for people who have never written
code in any deep learning framework (e.g, TensorFlow,Theano, Keras, DyNet).
The tutorials assumes working knowledge of core NLP problems: part-of-speech
tagging, language modeling, etc. It also assumes familiarity with neural
networks at the level of an intro AI class (such as one from the Russel and
Norvig book). Usually, these courses cover the basic backpropagation algorithm
on feed-forward neural networks, and make the point that they are chains of
compositions of linearities and non-linearities. This tutorial aims to get
you started writing deep learning code, given you have this prerequisite
knowledge.

Note these tutorials are about *models*, not data. For all of the models,
a few test examples are created with small dimensionality so you can see how
the weights change as it trains. If you have some real data you want to
try, you should be able to rip out any of the models from this notebook
and use them on it.

1. pytorch_tutorial.py
	Introduction to PyTorch
	https://tutorials.pytorch.kr/beginner/nlp/pytorch_tutorial.html

2. deep_learning_tutorial.py
	Deep Learning with PyTorch
	https://tutorials.pytorch.kr/beginner/nlp/deep_learning_tutorial.html

3. word_embeddings_tutorial.py
	Word Embeddings: Encoding Lexical Semantics
	https://tutorials.pytorch.kr/beginner/nlp/word_embeddings_tutorial.html

4. sequence_models_tutorial.py
	Sequence Models and Long Short-Term Memory Networks
	https://tutorials.pytorch.kr/beginner/nlp/sequence_models_tutorial.html

5. advanced_tutorial.py
	Advanced: Making Dynamic Decisions and the Bi-LSTM CRF
	https://tutorials.pytorch.kr/beginner/nlp/advanced_tutorial.html
