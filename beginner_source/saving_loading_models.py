# -*- coding: utf-8 -*-
"""
모델 저장하기 & 불러오기
=========================
**Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
  **번역**: `박정환 <http://github.com/9bow>`_, `김제필 <http://github.com/garlicvread>`_

이 문서에서는 PyTorch 모델을 저장하고 불러오는 다양한 방법을 제공합니다.
이 문서 전체를 다 읽는 것도 좋은 방법이지만, 필요한 사용 예의 코드만 참고하는
것도 고려해보세요.

모델을 저장하거나 불러올 때는 3가지의 핵심 함수와 익숙해질 필요가 있습니다:

1) `torch.save <https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save>`__:
   직렬화된 객체를 디스크에 저장합니다. 이 함수는 Python의
   `pickle <https://docs.python.org/3/library/pickle.html>`__ 을 사용하여 직렬화합니다.
   이 함수를 사용하여 모든 종류의 객체의 모델, Tensor 및 사전을 저장할 수 있습니다.

2) `torch.load <https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load>`__:
   `pickle <https://docs.python.org/3/library/pickle.html>`__\ 을 사용하여
   저장된 객체 파일들을 역직렬화하여 메모리에 올립니다. 이 함수는 데이터를 장치에 불러올
   때에도 사용됩니다.
   (`장치 간 모델 저장하기 & 불러오기 <#device>`__ 참고)

3) `torch.nn.Module.load_state_dict <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict>`__:
   역직렬화된 *state_dict* 를 사용하여 모델의 매개변수들을 불러옵니다.
   *state_dict* 에 대한 더 자세한 정보는 `state_dict가 무엇인가요?
   <#state-dict>`__ 를 참고하세요.



**목차:**

-  `state_dict가 무엇인가요? <#state-dict>`__
-  `추론(inference)를 위해 모델 저장하기 & 불러오기 <#inference>`__
-  `일반 체크포인트(checkpoint) 저장하기 & 불러오기 <#checkpoint>`__
-  `여러 개(multiple)의 모델을 하나의 파일에 저장하기 <#multiple>`__
-  `다른 모델의 매개변수를 사용하여 빠르게 모델 시작하기(warmstart) <#warmstart>`__
-  `장치(device)간 모델 저장하기 & 불러오기 <#device>`__

"""


######################################################################
# ``state_dict`` 가 무엇인가요?
# -------------------------------
#
# PyTorch에서 ``torch.nn.Module`` 모델의 학습 가능한 매개변수(예. 가중치와 편향)들은
# 모델의 매개변수에 포함되어 있습니다(model.parameters()로 접근합니다).
# *state_dict* 는 간단히 말해 각 계층을 매개변수 텐서로 매핑되는 Python 사전(dict)
# 객체입니다. 이 때, 학습 가능한 매개변수를 갖는 계층(합성곱 계층, 선형 계층 등)
# 및 등록된 버퍼들(batchnorm의 running_mean)만이 모델의 *state_dict* 에 항목을
# 가짐을 유의하시기 바랍니다. 옵티마이저 객체(``torch.optim``) 또한 옵티마이저의
# 상태 뿐만 아니라 사용된 하이퍼 매개변수(Hyperparameter) 정보가 포함된
# *state_dict* 를 갖습니다.
#
# *state_dict* 객체는 Python 사전이기 때문에 쉽게 저장하거나 갱신하거나 바꾸거나
# 되살릴 수 있으며, PyTorch 모델과 옵티마이저에 엄청난 모듈성(modularity)을 제공합니다.
#
# 예제:
# ^^^^^^^^
#
# :doc:`/beginner/blitz/cifar10_tutorial` 튜토리얼에서 사용한 간단한 모델의
# *state_dict* 를 살펴보도록 하겠습니다.
#
# .. code:: python
#
#    # 모델 정의
#    class TheModelClass(nn.Module):
#        def __init__(self):
#            super(TheModelClass, self).__init__()
#            self.conv1 = nn.Conv2d(3, 6, 5)
#            self.pool = nn.MaxPool2d(2, 2)
#            self.conv2 = nn.Conv2d(6, 16, 5)
#            self.fc1 = nn.Linear(16 * 5 * 5, 120)
#            self.fc2 = nn.Linear(120, 84)
#            self.fc3 = nn.Linear(84, 10)
#
#        def forward(self, x):
#            x = self.pool(F.relu(self.conv1(x)))
#            x = self.pool(F.relu(self.conv2(x)))
#            x = x.view(-1, 16 * 5 * 5)
#            x = F.relu(self.fc1(x))
#            x = F.relu(self.fc2(x))
#            x = self.fc3(x)
#            return x
#
#    # 모델 초기화
#    model = TheModelClass()
#
#    # 옵티마이저 초기화
#    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
#    # 모델의 state_dict 출력
#    print("Model's state_dict:")
#    for param_tensor in model.state_dict():
#        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
#    # 옵티마이저의 state_dict 출력
#    print("Optimizer's state_dict:")
#    for var_name in optimizer.state_dict():
#        print(var_name, "\t", optimizer.state_dict()[var_name])
#
# **출력:**
#
# ::
#
#    Model's state_dict:
#    conv1.weight     torch.Size([6, 3, 5, 5])
#    conv1.bias   torch.Size([6])
#    conv2.weight     torch.Size([16, 6, 5, 5])
#    conv2.bias   torch.Size([16])
#    fc1.weight   torch.Size([120, 400])
#    fc1.bias     torch.Size([120])
#    fc2.weight   torch.Size([84, 120])
#    fc2.bias     torch.Size([84])
#    fc3.weight   torch.Size([10, 84])
#    fc3.bias     torch.Size([10])
#
#    Optimizer's state_dict:
#    state    {}
#    param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
#


######################################################################
# 추론(inference)를 위해 모델 저장하기 & 불러오기
# ------------------------------------------------
#
# ``state_dict`` 저장하기 / 불러오기 (권장)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **저장하기:**
#
# .. code:: python
#
#    torch.save(model.state_dict(), PATH)
#
# **불러오기:**
#
# .. code:: python
#
#    model = TheModelClass(*args, **kwargs)
#    model.load_state_dict(torch.load(PATH))
#    model.eval()
#
# .. note::
#     PyTorch 버전 1.6에서는 ``torch.save`` 가 새로운 Zip파일-기반의 파일
#     포맷을 사용하도록 변경되었습니다. ``torch.load`` 는 예전 방식의 파일들을
#     읽어올 수 있도록 하고 있습니다. 어떤 이유에서든 ``torch.save`` 가 예전
#     방식을 사용하도록 하고 싶다면, ``_use_new_zipfile_serialization=False`` 을
#     kwarg로 전달하세요.
#
# 추론을 위해 모델을 저장할 때는 그 모델의 학습된 매개변수만 저장하면 됩니다.
# ``torch.save()`` 를 사용하여 모델의 *state_dict* 를 저장하는 것이 나중에 모델을
# 사용할 때 가장 유연하게 사용할 수 있는, 모델 저장 시 권장하는 방법입니다.
#
# PyTorch에서는 모델을 저장할 때 ``.pt`` 또는 ``.pth`` 확장자를 사용하는 것이
# 일반적인 규칙입니다.
#
# 추론을 실행하기 전에 반드시 ``model.eval()`` 을 호출하여 드롭아웃 및 배치
# 정규화를 평가 모드로 설정하여야 합니다. 이 과정을 거치지 않으면 일관성 없는
# 추론 결과가 출력됩니다.
#
# .. Note ::
#
#    ``load_state_dict()`` 함수에는 저장된 객체의 경로가 아닌, 사전 객체를
#    전달해야 하는 것에 유의하세요. 따라서 저장된 *state_dict* 를 ``load_state_dict()``
#    함수에 전달하기 전에 반드시 역직렬화를 해야 합니다. 예를 들어,
#    ``model.load_state_dict(PATH)`` 과 같은 식으로 사용하면 안됩니다.
#
# .. Note ::
#
#    만약 (검증 손실(validation loss) 결과에 따라) 가장 성능이 좋은 모델만 유지할
#    계획이라면, ``best_model_state = model.state_dict()`` 은 모델의 복사본이 아닌
#    모델의 현재 상태에 대한 참조(reference)만 반환한다는 사실을 잊으시면 안됩니다!
#    따라서 ``best_model_state`` 을 직렬화(serialize)하거나,
#    ``best_model_state = deepcopy(model.state_dict())`` 을 사용해야 합니다.
#    그렇지 않으면, 제일 좋은 성능을 내는 ``best_model_state`` 은 계속되는 학습 단계에서
#    갱신될 것입니다. 결과적으로, 최종 모델의 상태는 과적합(overfit)된 상태가 됩니다.
#
# 전체 모델 저장하기/불러오기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **저장하기:**
#
# .. code:: python
#
#    torch.save(model, PATH)
#
# **불러오기:**
#
# .. code:: python
#
#    # 모델 클래스는 어딘가에 반드시 선언되어 있어야 합니다.
#    model = torch.load(PATH)
#    model.eval()
#
# 이 저장하기/불러오기 과정은 가장 직관적인 문법을 사용하며 적은 양의
# 코드를 사용합니다. 이러한 방식으로 모델을 저장하는 것은 Python의
# `pickle <https://docs.python.org/3/library/pickle.html>`__ 모듈을 사용하여
# 전체 모듈을 저장하게 됩니다. 하지만 pickle은 모델 그 자체를 저장하지 않기 때문에
# 직렬화된 데이터가 모델을 저장할 때 사용한 특정 클래스 및 디렉토리 경로(구조)에
# 얽매인다는 것이 이 방식의 단점입니다. 대신에 클래스가 위치한 파일의 경로를
# 저장해두고, 불러오는 시점에 사용합니다. 이러한 이유 때문에, 만들어둔 코드를
# 다른 프로젝트에서 사용하거나 리팩토링 후에 다양한 이유로 동작하지 않을 수
# 있습니다.
#
# PyTorch에서는 모델을 저장할 때 ``.pt`` 또는 ``.pth`` 확장자를 사용하는 것이
# 일반적인 규칙입니다.
#
# 추론을 실행하기 전에는 반드시 ``model.eval()`` 을 호출하여 드롭아웃 및 배치
# 정규화를 평가 모드로 설정하여야 합니다. 이것을 하지 않으면 추론 결과가 일관성
# 없게 출력됩니다.
#
# TorchScript 포맷으로 모델 내보내기/가져오기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 훈련된 모델로 추론을 수행하는 일반적인 방법 중 하나는 `TorchScript <https://pytorch.org/docs/stable/jit.html>`__ 를 사용하는 것입니다.
# TorchScript는 파이썬 환경이나 C++와 같은 고성능 환경에서 실행할 수 있는
# 파이토치 모델의 중간 표현(IR; Intermediate Representation)입니다.
# TorchScript는 확장된 추론 및 배포에 권장되는 모델 형식이기도 합니다.
#
# .. note::
#    TorchScript 형식을 사용하면 모델 클래스를 정의하지 않고도 내보낸 모델을 읽어 오거나 추론을 실행할 수 있습니다.
#
# **Export:**
#
# .. code:: python
#
#    model_scripted = torch.jit.script(model) # TorchScript 형식으로 내보내기
#    model_scripted.save('model_scripted.pt') # 저장하기
#
# **Load:**
#
# .. code:: python
#
#    model = torch.jit.load('model_scripted.pt')
#    model.eval()
#
# 추론 실행 전, 드롭아웃 및 배치(batch) 정규화 레이어를 평가 모드로 설정하기 위해 ``model.eval()`` 을 호출해야
# 합니다. 이 호출 과정이 없으면 일관성 없는 추론 결과가 나타납니다.
#
# TorchScript에 대한 추가 정보는 전용
# `자습서 <https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html>`__ 에서 찾을 수 있습니다.
# `C++ 환경 <https://tutorials.pytorch.kr/advanced/cpp_export.html>`__ 문서를 참고하여 트레이싱(Tracing) 변환을
# 수행하는 방법과 C++ 환경에서 TorchScript 모듈을 실행하는 방법을 익힐 수 있습니다.



######################################################################
# 추론 / 학습 재개를 위해 일반 체크포인트(checkpoint) 저장하기 & 불러오기
# --------------------------------------------------------------------------
#
# 저장하기:
# ^^^^^^^^^^
#
# .. code:: python
#
#    torch.save({
#                'epoch': epoch,
#                'model_state_dict': model.state_dict(),
#                'optimizer_state_dict': optimizer.state_dict(),
#                'loss': loss,
#                ...
#                }, PATH)
#
# 불러오기:
# ^^^^^^^^^^
#
# .. code:: python
#
#    model = TheModelClass(*args, **kwargs)
#    optimizer = TheOptimizerClass(*args, **kwargs)
#
#    checkpoint = torch.load(PATH)
#    model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#    epoch = checkpoint['epoch']
#    loss = checkpoint['loss']
#
#    model.eval()
#    # - or -
#    model.train()
#
# 추론 또는 학습 재개를 위해 일반 체크포인트를 저장할 때는 반드시 모델의
# *state_dict* 보다 많은 것들을 저장해야 합니다. 모델이 학습을 하며 갱신되는
# 버퍼와 매개변수가 포함된 옵티마이저의 *state_dict* 도 함께 저장하는 것이
# 중요합니다. 그 외에도 마지막 에폭(epoch), 최근에 기록된 학습 손실, 외부
# ``torch.nn.Embedding`` 계층 등도 함께 저장합니다. 결과적으로, 이런 체크포인트는
# 종종 모델만 저장하는 것보다 2~3배 정도 커지게 됩니다.
#
# 여러가지를 함께 저장하려면, 사전(dictionary) 자료형으로 만든 후
# ``torch.save()`` 를 사용하여 직렬화합니다. PyTorch가 이러한 체크포인트를 저장할
# 때는 ``.tar`` 확장자를 사용하는 것이 일반적인 규칙입니다.
#
# 항목들을 불러올 때에는 먼저 모델과 옵티마이저를 초기화한 후, ``torch.load()``
# 를 사용하여 사전을 불러옵니다. 이후로는 저장된 항목들을 사전에 원하는대로 사전에
# 질의하여 쉽게 접근할 수 있습니다.
#
# 추론을 실행하기 전에는 반드시 ``model.eval()`` 을 호출하여 드롭아웃 및 배치
# 정규화를 평가 모드로 설정하여야 합니다. 이것을 하지 않으면 추론 결과가 일관성
# 없게 출력됩니다. 만약 학습을 계속하고 싶다면, ``model.train()`` 을 호출하여
# 학습 모드로 전환되도록 해야 합니다.
#


######################################################################
# 여러개(multiple)의 모델을 하나의 파일에 저장하기
# -------------------------------------------------------
#
# 저장하기:
# ^^^^^^^^^^
#
# .. code:: python
#
#    torch.save({
#                'modelA_state_dict': modelA.state_dict(),
#                'modelB_state_dict': modelB.state_dict(),
#                'optimizerA_state_dict': optimizerA.state_dict(),
#                'optimizerB_state_dict': optimizerB.state_dict(),
#                ...
#                }, PATH)
#
# 불러오기:
# ^^^^^^^^^^
#
# .. code:: python
#
#    modelA = TheModelAClass(*args, **kwargs)
#    modelB = TheModelBClass(*args, **kwargs)
#    optimizerA = TheOptimizerAClass(*args, **kwargs)
#    optimizerB = TheOptimizerBClass(*args, **kwargs)
#
#    checkpoint = torch.load(PATH)
#    modelA.load_state_dict(checkpoint['modelA_state_dict'])
#    modelB.load_state_dict(checkpoint['modelB_state_dict'])
#    optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
#    optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
#
#    modelA.eval()
#    modelB.eval()
#    # - or -
#    modelA.train()
#    modelB.train()
#
# GAN, Seq2Seq 또는 앙상블 모델과 같이 여러개의 여러개의 ``torch.nn.Modules`` 로
# 구성된 모델을 저장하는 경우에는 일반 체크포인트를 저장할 때와 같은 방식을
# 따릅니다. 즉, 각 모델의 *state_dict* 와 해당 옵티마이저를 사전으로 저장합니다.
# 앞에서 언급했던 것과 같이, 학습을 재개하는데 필요한 다른 항목들을 사전에 추가하여
# 저장할 수 있습니다.
#
# PyTorch가 이러한 체크포인트를 저장할 때는 ``.tar`` 확장자를 사용하는 것이
# 일반적인 규칙입니다.
#
# 항목들을 불러올 때에는 먼저 모델과 옵티마이저를 초기화한 후, ``torch.load()``
# 를 사용하여 사전을 불러옵니다. 이후로는 저장된 항목들을 사전에 원하는대로 사전에
# 질의하여 쉽게 접근할 수 있습니다.
#
# 추론을 실행하기 전에는 반드시 ``model.eval()`` 을 호출하여 드롭아웃 및 배치
# 정규화를 평가 모드로 설정하여야 합니다. 이것을 하지 않으면 추론 결과가 일관성
# 없게 출력됩니다. 만약 학습을 계속하고 싶다면, ``model.train()`` 을 호출하여
# 학습 모드로 설정해야 합니다.
#


######################################################################
# 다른 모델의 매개변수를 사용하여 빠르게 모델 시작하기(warmstart)
# --------------------------------------------------------------------
#
# 저장하기:
# ^^^^^^^^^^
#
# .. code:: python
#
#    torch.save(modelA.state_dict(), PATH)
#
# 불러오기:
# ^^^^^^^^^^
#
# .. code:: python
#
#    modelB = TheModelBClass(*args, **kwargs)
#    modelB.load_state_dict(torch.load(PATH), strict=False)
#
# 부분적으로 모델을 불러오거나, 모델의 일부를 불러오는 것은 전이학습 또는
# 새로운 복잡한 모델을 학습할 때 일반적인 시나리오입니다. 학습된 매개변수를
# 사용하면, 일부만 사용한다 하더라도 학습 과정을 빠르게 시작할 수 있고,
# 처음부터 시작하는 것보다 훨씬 빠르게 모델이 수렴하도록 도울 것입니다.
#
# 몇몇 키를 제외하고 *state_dict* 의 일부를 불러오거나, 적재하려는 모델보다
# 더 많은 키를 갖고 있는 *state_dict* 를 불러올 때에는 ``load_state_dict()``
# 함수에서 ``strict`` 인자를 **False** 로 설정하여 일치하지 않는 키들을
# 무시하도록 해야 합니다.
#
# 한 계층에서 다른 계층으로 매개변수를 불러오고 싶지만, 일부 키가 일치하지
# 않을 때에는 적재하려는 모델의 키와 일치하도록 *state_dict* 의 매개변수 키의
# 이름을 변경하면 됩니다.
#


######################################################################
# 장치(device)간 모델 저장하기 & 불러오기
# ----------------------------------------
#
# GPU에서 저장하고 CPU에서 불러오기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **저장하기:**
#
# .. code:: python
#
#    torch.save(model.state_dict(), PATH)
#
# **불러오기:**
#
# .. code:: python
#
#    device = torch.device('cpu')
#    model = TheModelClass(*args, **kwargs)
#    model.load_state_dict(torch.load(PATH, map_location=device))
#
# GPU에서 학습한 모델을 CPU에서 불러올 때는 ``torch.load()`` 함수의
# ``map_location`` 인자에 ``torch.device('cpu')`` 을 전달합니다.
# 이 경우에는 Tensor에 저장된 내용들은 ``map_location`` 인자를 사용하여 CPU 장치에
# 동적으로 재배치됩니다.
#
# GPU에서 저장하고 GPU에서 불러오기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **저장하기:**
#
# .. code:: python
#
#    torch.save(model.state_dict(), PATH)
#
# **불러오기:**
#
# .. code:: python
#
#    device = torch.device("cuda")
#    model = TheModelClass(*args, **kwargs)
#    model.load_state_dict(torch.load(PATH))
#    model.to(device)
#    # 모델에서 사용하는 input Tensor들은 input = input.to(device) 을 호출해야 합니다.
#
# GPU에서 학습한 모델을 GPU에서 불러올 때에는, 초기화된 ``model`` 에
# ``model.to(torch.device('cuda'))`` 을 호출하여 CUDA 최적화된 모델로 변환해야
# 합니다. 또한, 모델에 데이터를 제공하는 모든 입력에 ``.to(torch.device('cuda'))``
# 함수를 호출해야 합니다. ``my_tensor.to(device)`` 를 호출하면 GPU에 ``my_tensor``
# 의 복사본을 반환하기 때문에, Tensor를 직접 덮어써야 합니다:
# ``my_tensor = my_tensor.to(torch.device('cuda'))`` .
#
# CPU에서 저장하고 GPU에서 불러오기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **저장하기:**
#
# .. code:: python
#
#    torch.save(model.state_dict(), PATH)
#
# **불러오기:**
#
# .. code:: python
#
#    device = torch.device("cuda")
#    model = TheModelClass(*args, **kwargs)
#    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # 사용할 GPU 장치 번호를 선택합니다.
#    model.to(device)
#    # 모델에서 사용하는 input Tensor들은 input = input.to(device) 을 호출해야 합니다.
#
# CPU에서 학습한 모델을 GPU에서 불러올 때는 ``torch.load()`` 함수의
# ``map_location`` 인자에 *cuda:device_id* 을 설정합니다. 이렇게 하면 모델이 해당
# GPU 장치에 불러와집니다. 다음으로 ``model.to(torch.device('cuda'))`` 을 호출하여
# 모델의 매개변수 Tensor들을 CUDA Tensor들로 변환해야 합니다. 마지막으로 모든
# 모델 입력에 ``.to(torch.device('cuda'))`` 을 사용하여 CUDA 최적화된 모델을 위한
# 데이터로 만들어야 합니다. ``my_tensor.to(device)`` 를 호출하면 GPU에 ``my_tensor``
# 의 복사본을 반환합니다. 이 동작은 ``my_tensor`` 를 덮어쓰지 않기 때문에, Tensor를
# 직접 덮어써야 합니다: ``my_tensor = my_tensor.to(torch.device('cuda'))`` .
#
# ``torch.nn.DataParallel`` 모델 저장하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **저장하기:**
#
# .. code:: python
#
#    torch.save(model.module.state_dict(), PATH)
#
# **불러오기:**
#
# .. code:: python
#
#    # 사용할 장치에 불러옵니다.
#
# ``torch.nn.DataParallel`` 은 병렬 GPU 활용을 가능하게 하는 모델 래퍼(wrapper)입니다.
# ``DataParallel`` 모델을 범용적으로 저장하려면 ``model.module.state_dict()`` 을
# 사용하면 됩니다. 이렇게 하면 원하는 모든 장치에 원하는 방식으로 유연하게 모델을
# 불러올 수 있습니다.
#
