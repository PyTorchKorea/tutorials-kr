# -*- coding: utf-8 -*-
"""
TorchRL 기반 강화학습 (PPO) 튜토리얼
==================================================
**저자**: `Vincent Moens <https://github.com/vmoens>`_
**번역**: `박태은 <https://github.com/ptesogno>`_

이 튜토리얼에서는 PyTorch와 :py:mod:`torchrl` 을 사용해 모수적 정책망(parametric policy network)를 학습시키고
`OpenAI-Gym/Farama-Gymnasium control library <https://github.com/Farama-Foundation/Gymnasium>`__의
Inverted Pendulum 문제를 해결하는 방법을 설명합니다.

.. figure:: /_static/img/invpendulum.gif
   :alt: Inverted pendulum

   Inverted pendulum

핵심 학습 내용:

- TorchRL에서 환경을 생성하고 출력값을 변환하며, 해당 환경에서 데이터를 수집하는 방법
- :class:`~tensordict.TensorDict`를 사용해 여러 클래스가 서로 데이터를 주고받도록 만드는 방법
- TorchRL을 이용한 학습 루프(training loop) 구성의 기초:

  - 정책 변화도(policy gradient) 메서드에서 advantage 신호를 계산하는 방법
  - 확률적 신경망을 사용해 stochastic policy를 만드는 방법
  - 동적 replay buffer를 만들고 중복 없이 샘플링하는 방법

이 튜토리얼에서는 TorchRL의 핵심 구성 요소 6가지를 다룹니다.

* `environments <https://pytorch.org/rl/reference/envs.html>`__
* `transforms <https://pytorch.org/rl/reference/envs.html#transforms>`__
* `models (policy and value function) <https://pytorch.org/rl/reference/modules.html>`__
* `loss modules <https://pytorch.org/rl/reference/objectives.html>`__
* `data collectors <https://pytorch.org/rl/reference/collectors.html>`__
* `replay buffers <https://pytorch.org/rl/reference/data.html#replay-buffers>`__

"""

######################################################################
# Google Colab에서 실행 중이라면 아래 패키지를 먼저 설치해야 합니다.
#
# .. code-block:: bash
#
#    !pip3 install torchrl
#    !pip3 install gym[mujoco]
#    !pip3 install tqdm
#
# Proximal Policy Optimization (PPO)는 정책 변화도(policy-gradient) 알고리즘으로,
# 일정량의 데이터를 수집하고 즉시 사용해 정책(policy)이 근접성 제약(proximality constraint)을 적용하면서 
# 기대 보상(expected return)을 최대화하도록 학습합니다. PPO는 기초적인 정책 최적화 알고리즘인
# `REINFORCE <https://link.springer.com/content/pdf/10.1007/BF00992696.pdf>`_의
# 발전된 형태로 볼 수 있습니다. 자세한 내용은 아래 논문을 참고하세요.
# `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_
#
# PPO는 일반적으로 online, on-policy 강화학습 알고리즘 중에서
# 빠르고 효율적인 방법으로 평가됩니다. TorchRL은 PPO를 위한 손실 모듈을 제공하여
# 정책을 학습할 때마다 동일한 기능을 직접 다시 구현하는 대신
# 모듈을 그대로 사용해 문제 해결에 집중할 수 있게 합니다.
#
# 이해를 돕기 위해 :class:`~torchrl.objectives.ClipPPOLoss` 모듈에서 처리하는 손실 계산에 대해 간단히 설명하겠습니다.
# 알고리즘 흐름:
# 1. 일정 단계동안 환경 내에서 정책을 이용해 데이터를 샘플링합니다.
# 2. 수집한 batch 데이터의 무작위 부표본(sub-sample)에서 clipped REINFORCE loss를 사용해
# 주어진 횟수만큼 최적화 단계를 수행합니다.
# 3. 클리핑(clipping)은 손실에 보수적인 제한을 두어 비교적 낮은 추정값을
# 선호하게 만듭니다.
# 손실 수식은 아래와 같습니다.
#
# .. math::
#
#     L(s,a,\theta_k,\theta) = \min\left(
#     \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
#     g(\epsilon, A^{\pi_{\theta_k}}(s,a))
#     \right),
#
# 이 loss는 크게 두 부분으로 이루어져 있습니다. 첫 번째는 minimum 연산자의 첫 번째 항으로,
# 중요도 가중치(importance-weight)가 적용된 REINFORCE loss를 계산합니다.
# (현재 정책의 구성이 데이터 수집에 사용된 정책보다 뒤쳐져 있다는 점을 보완한
# REINFORCE loss를 예시로 들 수 있습니다.)
# minimum 연산자의 두 번째 항은 ratio 값이 특정 임계값보다 크거나 작을 때
# clipping을 적용한 손실과 유사합니다.
#
# 이 손실은 advantage가 양수든 음수든,
# 정책이 이전 형태에서 지나치게 크게 변하는
# 업데이트를 막습니다.
#
# 이 튜토리얼은 아래와 같이 구성되어 있습니다.
#
# 1. 먼저, 학습에 사용할 하이퍼파라미터셋을 정의합니다.
#
# 2. 다음으로 TorchRL의 래퍼와 변환을 사용해 환경 혹은 시뮬레이터를 생성할 것입니다.
#
# 3. 이후 손실 함수에 필수적인 정책 네트워크와 가치 모델(value model)을 설계합니다.
#    이 모듈은 손실 모듈을 구성하는 데 사용될 것입니다.
#
# 4. 다음으로 리플레이 버퍼와 데이터 로더를 생성합니다.
#
# 5. 마지막으로, 학습 루프를 실행하고 결과를 분석합니다.
#
# 이 튜토리얼 전반에서 :mod:`tensordict` 라이브러리를 사용합니다.
# :class:`~tensordict.TensorDict` 는 TorchRL의 공통 인터페이스 역할을 합니다.
# 이는 각 모듈이 어떤 데이터를 읽고 쓰는지를 추상화하며,  
# 특정 데이터 구조 자체보다 그 알고리즘에 더 집중할 수 있게 합니다.
#

import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

# sphinx_gallery_start_ignore

# TorchRL은 `__main__` 내부에서 ``~torchrl.envs.ParallelEnv`` 생성을 제한하는 spawn 방식을 선호하지만,
# 여기서는 코드 가독성을 위해 Google Colaboratory의 기본 spawn 방식인
# fork 방식을 사용합니다.
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

# sphinx_gallery_end_ignore

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

######################################################################
# 하이퍼파라미터 정의
# ----------------------
#
# 알고리즘에 사용할 하이퍼파라미터를 설정합니다. 사용 가능한 자원에 따라
# GPU 또는 다른 장치에서 정책을 실행할 수 있습니다.
# ``frame_skip``은 하나의 액션이 몇 프레임동안 반복 실행될지를
# 제어합니다. (한 번의 environment step이 ``frame_skip`` 개의 프레임을 반환하기 때문에)
# 프레임 수를 세는 다른 인자들도 이 값을 고려해 보정되어야 합니다.
#

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # 각 레이어의 셀 수. 즉, 출력의 차원 수
lr = 3e-4
max_grad_norm = 1.0

######################################################################
# 데이터 수집 매개변수
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 데이터를 수집할 때 ``frames_per_batch`` 매개변수를 정의해 각 배치의 크기를
# 결정할 수 있습니다. 사용할 수 있는 프레임 수(시뮬레이터와 상호작용하는 횟수 등) 또한 정의합니다.
# 일반적으로 강화학습 알고리즘의 목표는 환경과의 상호작용 측면에서 최대한 빠르게
# 문제를 해결하도록 학습하는 것입니다.
# 즉, ``total_frames``값이 낮을 수록 좋습니다.
#
frames_per_batch = 1000
# 완전한 학습을 위해 100만까지 프레임 수를 늘리세요.
total_frames = 50_000

######################################################################
# PPO 매개변수
# ~~~~~~~~~~~~~~
#
# 매 데이터 수집(혹은 배치 수집)마다 일정 수의 에폭동안 최적화를 수행하며,
# 매번 중첩된 학습 루프 안에서 방금 획득한 전체 데이터를 소비합니다.
# 여기서 ``sub_batch_size``는 위의 ``frames_per_batch``와 다릅니다.
# 수집기(collector)로부터 수집되고 ``frames_per_batch``에 의해 크기가 정의된 "데이터 배치"를 가지고 작업하며,
# 내부 학습 루프 동안 이를 더 작은 서브 배치(sub-batch)로 세분화한다는 점을 기억하세요.
# 이 서브 배치의 크기는 ``sub_batch_size``에 의해 제어됩니다.
#
sub_batch_size = 64  # 내부 루프 최적화 단계에서 현재 데이터로부터 수집된 서브 샘플의 수(cardinality)
num_epochs = 10  # 수집된 데이터 배치당 최적화 스텝 수
clip_epsilon = (
    0.2  # PPO 손실의 클리핑 값. 도입부의 수식을 참고하세요.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

######################################################################
# 환경 정의하기
# ---------------------
#
# 강화학습에서 *환경(environment)*은 보통 시뮬레이터나 제어 시스템을 지칭하는 용어입니다.
# Gymnasium(구 OpenAI Gym), DeepMind control suite 등을 포함한
# 다양한 라이브러리가 강화학습을 위한 시뮬레이션 환경을 제공합니다.
# 범용 라이브러리로서 TorchRL의 목표는 광범위한 RL 시뮬레이터에 대해 서로 교체 가능한 인터페이스를 제공하여
# 하나의 환경을 다른 환경으로 쉽게 바꿀 수 있도록 하는 것입니다.
# 예를 들어, 래핑된 gym 환경은 단 몇 글자만으로 생성할 수 있습니다
#

base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

######################################################################
# 이 코드에서 몇 가지 주의 깊게 볼 사항이 있습니다. 첫째, ``GymEnv``
# 클래스를 호출하여 환경을 생성했습니다. 만약 일반적인 내부 환경 매개변수를
# 변경해야 한다면(``GymEnv("Pendulum-v1", g=9.81)``과 같이 중력 가속도를 설정하는 등),
# 이러한 인수들을 변환 레이어가 아닌 ``GymEnv`` 커스텀 빌더로 직접 전달하면 됩니다.
# 대신 ``gym.make(env_name, **kwargs)``를 사용하여 gym 환경을 직접 생성하고,
# 이를 `GymWrapper` 클래스로 감싸는 것도 가능합니다.
#
# 둘째, 상태(state) 정보와 행동(action)이 위치할 ``device``를 지정했습니다.
# GymEnv`의 경우, 이 인자는 입력 액션과 관측된 상태가 저장될 디바이스만 제어할 뿐,
# 실제 시뮬레이션은 언제나 CPU에서 수행됩니다. 명시적으로 지정되지 않는 한, gym이 장치 내부(on-device) 실행을
# 지원하지 않기 때문입니다. 다른 라이브러리의 경우, 실행 장치를
# 제어할 수 있으며 가능한 한 저장 및 실행 백엔드를 일관되게 
# 유지하려 노력합니다.
#
# 변환(Transforms)
# ~~~~~~~~~~
#
# 정책을 위한 데이터를 준비하기 위해 몇 가지 변환(transforms)을 환경에 추가할 것입니다.
# Gym에서는 주로 래퍼(wrappers)를 통해 이를 달성합니다. TorchRL은 변환의 사용을 통해, 다른
# pytorch 도메인 라이브러리들과 더 유사한 다른 접근 방식을 취합니다. 
# 환경에 변환을 추가하려면, 단순히 이를 :class:`~torchrl.envs.transforms.TransformedEnv`
# 인스턴스로 감싸고 여기에 변환 시퀀스를 추가하면 됩니다. 변환된 환경은
# 감싸진 환경의 디바이스와 메타데이터를 상속받으며, 포함하고 있는 변환 시퀀스에 
# 따라 이들을 변환합니다.
#
# 정규화
# ~~~~~~~~~~~~~
#
# 가장 먼저 인코딩할 것은 정규화 변환입니다.
# 경험상, 데이터가 대략적으로 표준 가우시안 분포와 일치하도록
# 만드는 것이 바람직합니다. 이를 달성하기 위해
# 환경에서 일정 횟수의 무작위 스텝을 실행하고
# 이러한 관측치들의 요약 통계량을 계산할 것입니다.
#
# 다른 두 가지 변환도 추가할 것입니다. :class:`~torchrl.envs.transforms.DoubleToFloat` 변환은
# double 입력을 정책(policy)이 읽을 준비가 된 단일 정밀도(single-precision) 숫자로
# 변환합니다. :class:`~torchrl.envs.transforms.StepCounter` 변환은 환경이 종료되기 전까지의
# 스텝을 세는 데 사용됩니다. 이 측정값을 성능의 부가적인 측정 기준으로
# 사용할 것입니다.
#
# 나중에 보겠지만, TorchRL의 많은 클래스는 통신을 위해 :class:`~tensordict.TensorDict`에
# 의존합니다. 이를 몇 가지 추가적인 tensor 기능이 포함된 파이썬 딕셔너리로 생각할 수 있습니다.
# 실질적으로 이는 앞으로 다루게 될 많은 모듈에, 수신할 ``tensordict`` 내에서
# 어떤 키를 읽어야 하고(``in_keys``) 어떤 키에 써야 하는지(``out_keys``)를
# 알려주어야 함을 의미합니다. 보통 ``out_keys``가 생략되면, ``in_keys`` 항목들이
# in-place에서 업데이트된다고 가정합니다. 변환에서 관심을 두는
# 유일한 항목은 ``"observation"``으로 지칭되며, 변환 레이어들은 이 항목만을
# 수정하도록 지시받게 됩니다.
#

env = TransformedEnv(
    base_env,
    Compose(
        # 관측값 정규화
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

######################################################################
# 알아차렸을 수도 있겠지만, 정규화 레이어를 생성할 때 정규화 파라미터들을
# 설정하지 않았습니다. 이를 위해, :class:`~torchrl.envs.transforms.ObservationNorm`은
# 환경의 요약 통계량을 자동으로 수집할 수 있습니다.
#
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

######################################################################
# 이제 :class:`~torchrl.envs.transforms.ObservationNorm` 변환은 데이터를 정규화하는 데
# 사용될 위치와 스케일(scale) 값들로 채워졌습니다.
#
# 요약 통계량의 형태에 대해 간단한 기본 검증(sanity check)를 해봅시다.
#
print("normalization constant shape:", env.transform[0].loc.shape)

######################################################################
# 환경은 시뮬레이터와 변환들에 의해서만 정의되는 것이 아니라, 실행 과정 동안
# 무엇을 기대할 수 있는지 설명하는 일련의 메타데이터에 의해서도 정의됩니다.
# 효율성을 위해 TorchRL은 환경 명세(specs)에 있어서 꽤 엄격하지만,
# 여러분의 환경 명세가 적절한지는 쉽게 확인할 수 있습니다.
# 예시에서, :class:`~torchrl.envs.libs.gym.GymWrapper`와 이로부터 상속받는
# :class:`~torchrl.envs.libs.gym.GymEnv`는 이미 여러분의 환경에 맞는 적절한 명세를
# 설정하도록 처리해주므로, 이에 대해 신경 쓸 필요가 없을 것입니다.
#
# 그럼에도 불구하고, 변환된 환경의 명세를 살펴보면서 구체적인 예시를 확인해 봅시다.
# 살펴볼 명세는 세 가지가 있습니다. 환경에서 액션을 실행할 때 무엇을 기대할 수 있는지
# 정의하는 ``observation_spec``, 보상 도메인을 나타내는 ``reward_spec``, 그리고 마지막으로
# 환경이 단일 스텝을 실행하는 데 필요한 모든 것을 나타내는 ``input_spec``
# (이곳에 ``action_spec``이 포함)입니다.
#
print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

######################################################################
# func:`check_env_specs` 함수는 작은 롤아웃(rollout)을 실행하고 그 출력을 환경 명세와
# 비교합니다. 오류가 발생하지 않는다면, 명세가 올바르게 정의되었다고 확신할 수 있습니다.
#
check_env_specs(env)

######################################################################
# 재미 삼아, 간단한 무작위 롤아웃이 어떻게 생겼는지 확인해 봅시다.
# `env.rollout(n_steps)`를 호출하여 환경의 입력과 출력이 어떻게 구성되어 있는지
# 개요를 얻을 수 있습니다. 액션은 자동으로 action spec 도메인에서 추출되므로,
# 무작위 샘플러를 직접 설계하는 것에 대해 신경 쓸 필요가 없습니다.
#
# 일반적으로 각 스텝에서 RL 환경은 액션을 입력으로 받고, 관측치, 보상,
# done 상태를 출력합니다. 관측치는 복합적(composite)일 수 있으며, 이는 둘 이상의
# tensor로 구성될 수 있음을 의미합니다. 전체 관측치 세트가 출력 :class:`~tensordict.TensorDict`에
# 자동으로 패킹되므로 이는 TorchRL에서 문제가 되지 않습니다. 지정된 스텝 수 동안
# 롤아웃(예를 들어 일련의 환경 스텝 및 무작위 액션 생성하는 것)을 실행한 후,
# 이 궤적(trajectory) 길이와 일치하는 형태(shape)를 가진 :class:`~tensordict.TensorDict` 인스턴스를 얻게 됩니다.
#
rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

######################################################################
# 롤아웃 데이터는 ``torch.Size([3])``의 형태를 가지며, 이는 앞서 실행한
# 스텝 수와 일치합니다. ``"next"`` 항목은 현재 스텝 이후에 나오는 데이터를 가리킵니다.
# 대부분의 경우 시간 `t`에서의 ``"next"`` 데이터는 ``t+1``에서의 데이터와 일치하지만,
# 특정 변환(멀티 스텝 등)을 사용하는 경우에는 그렇지 않을 수도 있습니다.

#
# 정책(Policy)
# ------
#
# PPO는 탐색(exploration)을 처리하기 위해 확률적 정책(stochastic policy)을 활용합니다. 이는
# 신경망이 취할 액션에 해당하는 단일 값을 출력하는 대신, 분포의 매개변수(parameters)를
# 출력해야 함을 의미합니다.
#
# 데이터가 연속적(continuous)이므로, 액션 공간의 경계를 준수하기 위해 Tanh-Normal 분포를
# 사용할 것입니다. TorchRL은 이러한 분포를 제공하며, 신경망을 구축할 때 신경 써야 할 유일한 것은
# 정책이 작동할 수 있도록 올바른 개수의 매개변수(위치 혹은 평균, 그리고 스케일)를 출력하는 것입니다.
#
# .. math::
#
#     f_{\theta}(\text{observation}) = \mu_{\theta}(\text{observation}), \sigma^{+}_{\theta}(\text{observation})
#
# 여기서 추가되는 유일한 까다로운 점은 출력을 두 개의 동일한 부분으로 분할하고
# 두 번째 부분을 엄격하게 양수(positive)인 공간으로 매핑하는 것입니다.
#
# 아래 세 단계로 정책을 설계합니다.
#
# 1. ``D_obs`` -> ``2 * D_action`` 신경망을 정의합니다. 실제로 ``loc`` (mu)와 ``scale`` (sigma)은 모두 ``D_action`` 차원을 가집니다.
#
# 2. :class:`~tensordict.nn.distributions.NormalParamExtractor`를 추가하여 위치와 스케일을 추출합니다 (예를 들어, 입력을 두 개의 동일한 부분으로 분할하고 스케일 매개변수에 양수 변환을 적용합니다).

#
# 3. 이 분포를 생성하고 이로부터 샘플링할 수 있는 확률적 :class:`~tensordict.nn.TensorDictModule`을 생성합니다.
#

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

######################################################################
# 정책이 ``tensordict`` 데이터 캐리어를 통해 환경과 "대화"할 수 있도록 하기 위해
# ``nn.Module``을 :class:`~tensordict.nn.TensorDictModule`로 감쌉니다.
# 이 클래스는 제공된 ``in_keys``를 단순히 읽고 지정된 ``out_keys`` 위치에
# 출력값을 in-place에 작성합니다.
#
policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

######################################################################
# 이제 가우시안 분포의 위치와 스케일로부터 분포를 빌드해야 합니다.
# 이를 위해 :class:`~torchrl.modules.tensordict_module.ProbabilisticActor`
# 클래스가 위치와 스케일 매개변수로부터 :class:`~torchrl.modules.TanhNormal`을 빌드하도록
# 지시합니다. 환경 명세(specs)에서 수집한 이 분포의 최솟값과 최댓값 또한
# 함께 제공합니다.
#
# :class:`~torchrl.modules.TanhNormal` 분포 생성자가 ``loc``와 ``scale`` 키워드
# 인자를 요구하기 때문에, ``in_keys``의 이름(따라서 위 :class:`~tensordict.nn.TensorDictModule`에서
# 지정한 ``out_keys``의 이름)을 사용자가 원하는 임의의 값으로 설정할 수는 없습니다.
# 하지만 :class:`~torchrl.modules.tensordict_module.ProbabilisticActor`는
# 키-값 쌍이 사용될 모든 키워드 인자에 대해 어떤 ``in_key`` 문자열을 사용해야 하는지 나타내는
# ``Dict[str, str]`` 형태의 ``in_keys``도 허용합니다.
#
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    # 중요도 가중치의 분자에 사용할 로그 확률(log-prob)이 필요합니다.
)

######################################################################
# 가치 네트워크(Value network)
# -------------
#
# 가치 네트워크는 추론 시점에는 사용되지 않지만, PPO 알고리즘의 매우 중요한 구성 요소입니다.
# 이 모듈은 관측치를 읽고 이어지는 궤적이 줄어든 반환값의 추정치를 반환합니다.
# 이를 통해 학습 중에 즉석에서 학습되는 어떤 유틸리티 추정치에 의존함으로써
# 학습 비용을 균등하게 분산할 수 있습니다.
# 가치 네트워크는 정책과 동일한 구조를 공유하지만,
# 단순함을 위해 독립적인 별도의 매개변수 집합을 할당합니다.
#
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

######################################################################
# 정책 및 가치 모듈을 테스트해 봅시다. 앞서 언급했듯,
# :class:`~tensordict.nn.TensorDictModule`을 사용하면 어떤 정보를 읽고
# 어디에 기록해야 하는지 알고 있기 때문에, 환경의 출력을 직접 읽어서
# 이러한 모듈을 실행하는 것이 가능해집니다.
#
print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

######################################################################
# 데이터 수집기(Data collector)
# --------------
#
# TorchRL은 일련의 `DataCollector 클래스들 <https://pytorch.org/rl/reference/collectors.html>`__을 제공합니다.
# 간단히 말해서, 이 클래스들은 세 가지 연산을 수행합니다: 환경을 리셋하고,
# 최신 관측치가 주어졌을 때 액션을 계산하며, 환경에서 한 스텝을 실행하고,
# 환경이 중지 신호를 보낼 때까지(또는 done 상태에 도달할 때까지) 마지막 두 단계를 반복합니다.
#
# 이들은 각 반복(iteration)마다 얼마나 많은 프레임을 수집할지(``frames_per_batch`` 매개변수를 통해),
# 언제 환경을 리셋할지(``max_frames_per_traj`` 인수를 통해),
# 어떤 ``device``에서 정책을 실행해야 할지 등을 제어할 수 있게 해줍니다.
# 또, 배치화(batched) 및 다중 프로세스화(multiprocessed)된 환경에서 효율적으로 작동하도록 설계되었습니다.
#
# 가장 단순한 데이터 수집기는 :class:`~torchrl.collectors.collectors.SyncDataCollector`입니다.
# 이는 지정된 길이의 데이터 배치를 얻기 위해 사용할 수 있는 반복자(iterator)이며,
# 총 프레임 수(``total_frames``)가 수집되면 중지됩니다.
# 다른 데이터 수집기들(:class:`~torchrl.collectors.collectors.MultiSyncDataCollector` 및
# :class:`~torchrl.collectors.collectors.MultiaSyncDataCollector`)은 다중 프로세스화된
# 워커(workers) 세트에 대해 동기 및 비동기 방식으로 동일한 연산을 실행할 것입니다.
#
# 이전의 정책 및 환경과 마찬가지로, 데이터 수집기는 ``frames_per_batch``와
# 일치하는 총 원소 수를 가진 :class:`~tensordict.TensorDict` 인스턴스들을 반환할 것입니다.
# 학습 루프에 데이터를 전달하기 위해 :class:`~tensordict.TensorDict`를 사용하면, 롤아웃 내용의
# 실제 구체적인 특성에 100% 개의치 않는 데이터 로딩 파이프라인을 작성할 수 있습니다.
#
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

######################################################################
# 리플레이 버퍼(Replay buffer)
# -------------
#
# 리플레이 버퍼는 오프폴리시(off-policy) RL 알고리즘의 흔한 빌딩 블록입니다.
# on-policy 컨텍스트에서 리플레이 버퍼는 데이터 배치가 수집될 때마다 새로 채워지며,
# 그 데이터는 특정 에폭 수 동안 반복적으로 소비됩니다.
#
# TorchRL의 리플레이 버퍼는 버퍼의 구성 요소인 저장소(storage), 작성기(writer),
# 샘플러(sampler) 및 가용한 몇 가지 변환(transforms)을 인자로 받는 공통 컨테이너인
# :class:`~torchrl.data.ReplayBuffer`를 사용하여 빌드됩니다.
# 리플레이 버퍼의 용량을 나타내는 저장소만 필수적입니다.
# 또, 한 에폭 내에서 동일한 아이템이 여러 번 샘플링되는 것을 방지하기 위해
# 중복 없는 샘플러를 지정합니다.
# PPO에서 리플레이 버퍼를 사용하는 것이 필수적인 것은 아니며 수집된 배치에서 직접
# 서브 배치를 샘플링할 수도 있지만, 이 클래스들을 사용하면 내부 학습 루프를
# 재현 가능한 방식으로 쉽게 구축할 수 있습니다.
#

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

######################################################################
# 손실 함수
# -------------
#
# PPO 손실은 편의를 위해 :class:`~torchrl.objectives.ClipPPOLoss` 클래스를 사용하여
# TorchRL에서 직접 가져올 수 있습니다. 이것이 PPO를 사용하는 가장 쉬운 방법입니다.
# PPO의 수학적 연산과 그에 수반되는 제어 흐름을 내부로 숨겨줍니다.
#
# PPO는 어떤 "어드밴티지 추정(advantage estimation)"이 계산되어야 합니다.
# 요약하자면, advantage는 편향과 분산의 트레이드오프(bias / variance tradeoff)를
# 처리하면서 반환값(return value)에 대한 기댓값을 반영하는 값입니다.
# advantage를 계산하려면, 단순히 (1) 가치 연산자(value operator)를 활용하는
# 어드밴티지 모듈을 구축하고, (2) 각 에폭 전에 각 데이터 배치를 이 모듈에
# 통과시키기만 하면 됩니다.
# GAE 모듈은 입력 ``tensordict``를 새로운 ``"advantage"`` 및 ``"value_target"``
# 항목으로 업데이트합니다.
# ``"value_target"``은 입력 관측치에 대해 가치 네트워크가 나타내야 하는 경험적
# 가치(empirical value)를 나타내는 변화도가 없는 tensor입니다.
# 이 두 가지 모두 :class:`~torchrl.objectives.ClipPPOLoss`에 의해 정책 및 가치
# 손실을 반환하는 데 사용됩니다.
#

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # 이 키들은 기본적으로 일치하지만 완벽성을 위해 설정해 줍니다.
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

######################################################################
# 학습 루프
# -------------
# 이제 학습 루프를 코딩하는 데 필요한 모든 요소를 갖추었습니다.
# 단계는 다음과 같습니다.
#
# * 데이터 수집
#
#   * advantage 계산
#
#     * 수집된 데이터를 루프하며 손실 값 계산
#     * 역전파 (Back propagate)
#     * 최적화
#     * 반복
#
#   * 반복
#
# * 반복
#


logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# 수집기가 수집하도록 설계된 총 프레임 수에 도달할 때까지 수집기를 반복합니다.
for i, tensordict_data in enumerate(collector):
    # 이제 작업할 데이터 배치를 확보했습니다. 이것으로부터 무언가를 학습해 봅시다.
    for _ in range(num_epochs):
        # PPO를 작동시키려면 "advantage" 신호가 필요합니다.
        # 내부 루프에서 업데이트되는 가치 네트워크에 그 값이 의존하므로,
        # 매 에폭마다 이를 다시 계산합니다.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # 최적화: 역전파, 그래디언트 클리핑 및 최적화 단계
            loss_value.backward()
            # 이것이 엄격하게 필수적인 것은 아니지만, 그래디언트 놈(norm)을
            # 유계 상태로 유지하는 것이 좋습니다.
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # 데이터 배치가 10번 쌓일 때마다 정책을 평가합니다.
        # 평가는 다소 간단합니다. 지정된 스텝 수(``env`` 수평선인 1000스텝) 동안
        # 탐색 없이(액션 분포의 기댓값을 취함) 정책을 실행합니다.
        # ``env``의 ``rollout`` 메서드는 정책을 인자로 받을 수 있습니다.
        # 그러면 각 스텝에서 이 정책을 실행하게 됩니다.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # 학습된 정책으로 롤아웃을 실행합니다.
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # 학습률 스케줄러(learning rate scheduler)도 사용하고 있습니다. 변화도 클리핑과 마찬가지로,
    # 있으면 좋지만 PPO가 작동하는 데 필수적인 것은 아닙니다.
    scheduler.step()

######################################################################
# 결과
# -------
#
# 100만 스텝 한도에 도달하기 전에, 알고리즘은 최대 스텝 카운트인
# 1000스텝에 도달해야 하며, 이는 궤적이 폐기(truncated)되기 전 
# 최대 스텝 수입니다.
#
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()

######################################################################
# 결론 및 다음 단계
# -------------------------
#
# 이 튜토리얼에서 다음 내용을 학습했습니다.
#
# 1. :py:mod:`torchrl`을 사용하여 환경을 생성하고 커스텀하는 방법
# 2. 모델과 손실 함수를 작성하는 방법
# 3. 전형적인 학습 루프를 설정하는 방법
#
# 이 튜토리얼을 바탕으로 조금 더 실험해보고 싶다면, 다음과 같은 수정을 적용해 볼 수 있습니다.
#
# * 효율성 측면에서, 데이터 수집 속도를 높이기 위해 여러 시뮬레이션을
#   병렬로 실행할 수 있습니다.
#   자세한 내용은 :class:`~torchrl.envs.ParallelEnv`를 참조하세요.
#
# *  로깅 측면에서는, 역진자가 움직이는 모습을 시각적으로 렌더링하기 위해
#   렌더링을 요청한 후 환경에 :class:`torchrl.record.VideoRecorder` 변환을
#   추가할 수 있습니다.
#   자세한 내용은 :py:mod:`torchrl.record`를 참조하세요.
#
