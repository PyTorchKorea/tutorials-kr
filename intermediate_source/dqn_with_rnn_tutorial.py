# -*- coding: utf-8 -*-

"""
순환 DQN: 순환 정책 학습하기
==========================================

**저자**: `Vincent Moens <https://github.com/vmoens>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` 배울 내용
       :class-card: card-prerequisites

       * TorchRL에서 액터에 RNN을 통합하는 방법
       * 메모리 기반 정책을 리플레이 버퍼 및 손실 모듈과 함께 사용하는 방법

    .. grid-item-card:: :octicon:`list-unordered;1em;` 사전 준비 사항
       :class-card: card-prerequisites

       * PyTorch v2.0.0
       * gym[mujoco]
       * tqdm
"""

#########################################################################
# 개요
# --------
#
# 메모리 기반 정책(policy)은 관측이 부분적으로만 가능한 경우뿐만 아니라
# 의사결정을 위해 시간 차원을 고려해야 하는 경우에도 매우 중요합니다.
#
# 순환 신경망(Recurrent Neural Network)은 오랫동안 메모리 기반 정책에 널리 사용되어
# 왔습니다. 핵심 아이디어는 두 연속 단계(step) 사이에 순환 상태(recurrent state)를
# 메모리에 유지하고, 현재 관측과 함께 정책의 입력으로 사용하는 것입니다.
#
# 이 튜토리얼에서는 TorchRL을 사용하여 정책에 RNN을 통합하는 방법을 보여줍니다.
#
# 핵심 학습 내용:
#
# - TorchRL에서 액터에 RNN 통합하기
# - 메모리 기반 정책을 리플레이 버퍼 및 손실 모듈과 함께 사용하기
#
# TorchRL에서 RNN을 사용하는 핵심 아이디어는 TensorDict를 한 단계에서 다음 단계로
# 은닉 상태(hidden state)를 전달하는 데이터 운반체로 사용하는 것입니다. 이전
# 순환 상태를 현재 TensorDict에서 읽고, 현재 순환 상태를 다음 상태의
# TensorDict에 기록하는 정책을 구성합니다.
#
# .. figure:: /_static/img/rollout_recurrent.png
#    :alt: 순환 정책을 사용한 데이터 수집
#
# 이 그림에서 보듯이 환경은 TensorDict에 0으로 초기화된 순환 상태를 채우고,
# 정책은 이를 관측과 함께 읽어 행동과 다음 단계에 사용할 순환 상태를
# 생성합니다.
# :func:`~torchrl.envs.utils.step_mdp` 함수가 호출되면 다음 상태의 순환 상태가
# 현재 TensorDict로 가져옵니다. 이것이 실제로 어떻게 구현되는지
# 살펴보겠습니다.

######################################################################
# Google Colab에서 실행하는 경우 다음 의존성을 설치해야 합니다.
#
# .. code-block:: bash
#
#    !pip3 install torchrl
#    !pip3 install gym[mujoco]
#    !pip3 install tqdm
#
# 설정
# -----
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
from torch import multiprocessing

# TorchRL은 spawn 메소드를 선호하며, ``~torchrl.envs.ParallelEnv`` 생성을
# `__main__` 메소드 호출 내부로 제한하지만, 코드의 가독성을 위해 fork로 전환합니다.
# fork는 Google Colaboratory에서도 기본 시작 방식입니다.
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

# sphinx_gallery_end_ignore

import torch
import tqdm
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    ExplorationType,
    GrayScale,
    InitTracker,
    ObservationNorm,
    Resize,
    RewardScaling,
    set_exploration_type,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, EGreedyModule, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

######################################################################
# 환경
# -----------
#
# 먼저 환경을 구성합니다. 이를 통해 문제를 정의하고 그에 맞는 정책 네트워크를
# 구성할 수 있습니다. 이 튜토리얼에서는 CartPole gym 환경의 픽셀 관측을 사용하는 단일 환경
# 인스턴스를 사용하며, 몇 가지 사용자 정의 변환(transform)을 적용합니다.
# 그레이스케일 변환, 84x84 크기 변경, 보상 스케일링, 관측 정규화 등을 수행합니다.
#
# .. note::
#   :class:`~torchrl.envs.transforms.StepCounter` 변환은 보조적입니다. CartPole
#   태스크의 목표는 궤적(trajectory)을 가능한 한 길게 만드는 것이므로, 단계를 세는 것이
#   정책의 성능을 추적하는 데 도움이 됩니다.
#
# 이 튜토리얼의 목적에서 중요한 두 가지 변환이 있습니다.
#
# - :class:`~torchrl.envs.transforms.InitTracker` 는
#   :meth:`~torchrl.envs.EnvBase.reset` 호출을 표시하여 TensorDict에
#   ``"is_init"`` 불리언 마스크를 추가합니다. 이를 통해 RNN 은닉 상태를
#   초기화해야 하는 단계를 추적합니다.
# - :class:`~torchrl.envs.transforms.TensorDictPrimer` 변환은 좀 더 기술적입니다.
#   RNN 정책을 사용하는 데 반드시 필요하지는 않습니다. 그러나 환경(및 이후의
#   수집기)에 추가 키가 필요함을 알려줍니다. 추가되면 ``env.reset()`` 호출 시
#   프라이머에 지정된 항목이 0으로 초기화된 텐서(Tensor)로 채워집니다. 이 텐서가
#   정책에 필요하다는 것을 알고 있으므로 수집기는 수집 과정에서 이를 전달합니다.
#   결국 은닉 상태를 리플레이 버퍼에 저장하게 되며, 이는 손실 모듈에서
#   RNN 연산의 부트스트랩 계산에 도움이 됩니다(그렇지 않으면 0으로 초기화됩니다).
#   요약하면 이 변환을 포함하지 않아도 정책 학습에 큰 영향은 없지만, 수집된
#   데이터와 리플레이 버퍼에서 순환 키가 사라지게 되어 학습이 다소 최적에 미치지
#   못할 수 있습니다.
#   다행히 :class:`~torchrl.modules.LSTMModule` 은 이 변환을 자동으로
#   생성하는 헬퍼 메소드를 제공하므로, 모듈을 구성한 후에 사용하면 됩니다.
#

env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True, device=device),
    Compose(
        ToTensorImage(),
        GrayScale(),
        Resize(84, 84),
        StepCounter(),
        InitTracker(),
        RewardScaling(loc=0.0, scale=0.1),
        ObservationNorm(standard_normal=True, in_keys=["pixels"]),
    ),
)

######################################################################
# 항상 그렇듯이 정규화 상수를 수동으로 초기화해야 합니다.
#
env.transform[-1].init_stats(1000, reduce_dim=[0, 1, 2], cat_dim=0, keep_dims=[0])
td = env.reset()

######################################################################
# 정책
# ------
#
# 정책은 3개의 구성 요소로 이루어집니다. :class:`~torchrl.modules.ConvNet`
# 백본, :class:`~torchrl.modules.LSTMModule` 메모리 계층, 그리고 LSTM 출력을
# 행동 가치(action value)에 매핑하는 얕은 :class:`~torchrl.modules.MLP` 블록입니다.
#
# 합성곱 네트워크
# ~~~~~~~~~~~~~~~~~~~~~
#
# 출력을 크기 64의 벡터로 압축하는 :class:`torch.nn.AdaptiveAvgPool2d` 를 포함한
# 합성곱 네트워크를 구성합니다. :class:`~torchrl.modules.ConvNet` 이 이를
# 지원합니다.
#

feature = Mod(
    ConvNet(
        num_cells=[32, 32, 64],
        squeeze_output=True,
        aggregator_class=nn.AdaptiveAvgPool2d,
        aggregator_kwargs={"output_size": (1, 1)},
        device=device,
    ),
    in_keys=["pixels"],
    out_keys=["embed"],
)
######################################################################
# 출력 벡터의 크기를 얻기 위해 첫 번째 모듈을 데이터 배치에 대해 실행합니다.
#
n_cells = feature(env.reset())["embed"].shape[-1]

######################################################################
# LSTM 모듈
# ~~~~~~~~~~~
#
# TorchRL은 코드에 LSTM을 통합하기 위한 전용 :class:`~torchrl.modules.LSTMModule`
# 클래스를 제공합니다. 이 클래스는 :class:`~tensordict.nn.TensorDictModuleBase` 의
# 하위 클래스로, ``in_keys`` 와 ``out_keys`` 집합을 가지고 있어 모듈 실행 중
# 읽고 쓸/갱신할 값을 나타냅니다. 이 클래스는 생성을 쉽게 하기 위해
# 사전 정의된 기본값을 가지고 있습니다.
#
# .. note::
#   *사용 제한 사항*: 이 클래스는 드롭아웃(Dropout)이나 다중 계층 LSTM 등
#   대부분의 LSTM 기능을 지원합니다.
#   그러나 TorchRL의 규칙을 준수하기 위해 이 LSTM은 ``batch_first``
#   속성이 ``True`` 로 설정되어야 하며, 이는 PyTorch의 기본값이 **아닙니다**. 그러나
#   :class:`~torchrl.modules.LSTMModule` 은 이 기본 동작을 변경하므로
#   기본 호출만으로 충분합니다.
#
#   또한 LSTM의 ``bidirectional`` 속성이 ``True`` 로 설정되면 온라인 환경에서
#   사용할 수 없으므로, 기본값 그대로 사용합니다.
#

lstm = LSTMModule(
    input_size=n_cells,
    hidden_size=128,
    device=device,
    in_key="embed",
    out_key="embed",
)

######################################################################
# LSTM 모듈 클래스의 in_keys와 out_keys를 살펴보겠습니다.
print("in_keys", lstm.in_keys)
print("out_keys", lstm.out_keys)

######################################################################
# 이 값에는 in_key(및 out_key)로 지정한 키와 순환 키 이름이 포함되어
# 있습니다. out_keys 앞에는 "next" 접두사가 붙어 있어 "next" TensorDict에
# 기록해야 함을 나타냅니다.
# 이 규칙(in_keys/out_keys 인자를 전달하여 재정의할 수 있음)을 사용하면
# :func:`~torchrl.envs.utils.step_mdp` 호출 시 순환 상태가 루트 TensorDict로
# 이동하여, 다음 호출에서 RNN이 이를 사용할 수 있게 됩니다(도입부의 그림 참조).
#
# 앞서 언급한 대로 순환 상태가 버퍼에 전달되도록 환경에 선택적 변환을
# 하나 더 추가해야 합니다.
# :meth:`~torchrl.modules.LSTMModule.make_tensordict_primer` 메소드가
# 이를 정확히 수행합니다.
#
env.append_transform(lstm.make_tensordict_primer())

######################################################################
# 프라이머를 추가한 후 환경을 출력하여 모든 것이 올바른지 확인합니다.
print(env)

######################################################################
# MLP
# ~~~
#
# 정책에 사용할 행동 가치를 나타내기 위해 단일 계층 MLP를 사용합니다.
#
mlp = MLP(
    out_features=2,
    num_cells=[
        64,
    ],
    device=device,
)
######################################################################
# 편향(bias)을 0으로 채웁니다.

mlp[-1].bias.data.fill_(0.0)
mlp = Mod(mlp, in_keys=["embed"], out_keys=["action_value"])

######################################################################
# Q-값을 사용한 행동 선택
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 정책의 마지막 부분은 Q-값 모듈입니다.
# Q-값 모듈 :class:`~torchrl.modules.tensordict_module.QValueModule` 은
# MLP가 생성한 ``"action_values"`` 키를 읽고, 가장 높은 값을 가진 행동을
# 선택합니다.
# 행동 공간(action space)만 지정하면 되며, 문자열 또는 action-spec을 전달하여
# 지정할 수 있습니다. 이를 통해 범주형(Categorical, 때때로 "sparse"라고도 함)
# 인코딩 또는 원-핫(One-Hot) 버전을 사용할 수 있습니다.
#
qval = QValueModule(spec=env.action_spec)

######################################################################
# .. note::
#   TorchRL은 래퍼 클래스 :class:`torchrl.modules.QValueActor` 도 제공합니다.
#   이 클래스는 여기서 명시적으로 하는 것처럼 모듈을 Sequential과
#   :class:`~torchrl.modules.tensordict_module.QValueModule` 로 감쌉니다.
#   이렇게 하는 것에 큰 이점은 없고 과정이 덜 투명하지만, 최종 결과는
#   여기서 수행하는 것과 유사합니다.
#
# 이제 :class:`~tensordict.nn.TensorDictSequential` 로 구성 요소를 조합할 수
# 있습니다.
#
stoch_policy = Seq(feature, lstm, mlp, qval)

######################################################################
# DQN은 결정적(deterministic) 알고리즘이므로 탐색(exploration)이 매우 중요합니다.
# 초기값 0.2에서 점진적으로 0으로 감소하는 :math:`\epsilon`-탐욕(greedy) 정책을
# 사용합니다.
# 이 감소는 :meth:`~torchrl.modules.EGreedyModule.step` 호출을 통해
# 이루어집니다(아래 학습 루프 참조).
#
exploration_module = EGreedyModule(
    annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2
)
stoch_policy = Seq(
    stoch_policy,
    exploration_module,
)

######################################################################
# 손실에 모델 사용하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 지금까지 구성한 모델은 순차적 설정에서 사용하기에 적합합니다.
# 그러나 :class:`torch.nn.LSTM` 클래스는 GPU 장치에서 RNN 시퀀스를 더 빠르게
# 실행하기 위해 cuDNN 최적화 백엔드를 사용할 수 있습니다. 학습 루프를
# 가속할 수 있는 기회를 놓치지 않겠습니다.
# 이를 위해 손실에서 사용할 때 LSTM 모듈을 "recurrent-mode"로 실행하도록
# 지정하면 됩니다.
# 일반적으로 LSTM 모듈의 복사본 2개를 갖게 되므로,
# :meth:`~torchrl.modules.LSTMModule.set_recurrent_mode` 메소드를 호출하여
# 입력 데이터가 순차적임을 처리하는 새 인스턴스(가중치 공유)를 반환합니다.
#
policy = Seq(feature, lstm.set_recurrent_mode(True), mlp, qval)

######################################################################
# 아직 초기화되지 않은 매개변수(parameter)가 몇 개 있으므로
# 옵티마이저(Optimizer) 등을 생성하기 전에 초기화해야 합니다.
#
policy(env.reset())

######################################################################
# DQN 손실
# --------
#
# DQN 손실(loss)에는 정책과 행동 공간을 전달해야 합니다.
# 이것이 중복으로 보일 수 있지만, :class:`~torchrl.objectives.DQNLoss` 와
# :class:`~torchrl.modules.tensordict_module.QValueModule` 클래스가
# 호환되면서도 서로 강하게 의존하지 않도록 하기 위해 중요합니다.
#
# Double-DQN을 사용하기 위해 ``delay_value`` 인자를 요청하여
# 타겟 네트워크로 사용할 미분 불가능한 네트워크 매개변수 복사본을 생성합니다.
loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)

######################################################################
# Double DQN을 사용하고 있으므로 타겟 매개변수를 갱신해야 합니다.
# :class:`~torchrl.objectives.SoftUpdate` 인스턴스를 사용하여 이 작업을
# 수행합니다.
#
updater = SoftUpdate(loss_fn, eps=0.95)

optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

######################################################################
# 수집기와 리플레이 버퍼
# ---------------------------
#
# 가장 단순한 데이터 수집기를 구성합니다. 총 백만 프레임으로 알고리즘을
# 학습하며, 한 번에 50 프레임씩 버퍼를 확장합니다. 버퍼는 50 단계의
# 궤적 2만 개를 저장하도록 설계됩니다.
# 각 최적화 단계(데이터 수집당 16회)에서 버퍼로부터 4개 항목을 추출하여
# 총 200개의 전이(transition)를 처리합니다.
# 데이터를 디스크에 유지하기 위해 :class:`~torchrl.data.replay_buffers.LazyMemmapStorage`
# 스토리지를 사용합니다.
#
# .. note::
#   효율성을 위해 여기서는 수천 번의 반복만 실행합니다. 실제 환경에서는
#   총 프레임 수를 100만으로 설정해야 합니다.
#
collector = SyncDataCollector(env, stoch_policy, frames_per_batch=50, total_frames=200, device=device)
rb = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(20_000), batch_size=4, prefetch=10
)

######################################################################
# 학습 루프
# -------------
#
# 진행 상황을 추적하기 위해 50번의 데이터 수집마다 환경에서 정책을 한 번씩
# 실행하고, 학습 후 결과를 도식화합니다.
#

utd = 16
pbar = tqdm.tqdm(total=1_000_000)
longest = 0

traj_lens = []
for i, data in enumerate(collector):
    if i == 0:
        print(
            "Let us print the first batch of data.\nPay attention to the key names "
            "which will reflect what can be found in this data structure, in particular: "
            "the output of the QValueModule (action_values, action and chosen_action_value),"
            "the 'is_init' key that will tell us if a step is initial or not, and the "
            "recurrent_state keys.\n",
            data,
        )
    pbar.update(data.numel())
    # 평탄화되지 않은 데이터를 전달하는 것이 중요합니다
    rb.extend(data.unsqueeze(0).to_tensordict().cpu())
    for _ in range(utd):
        s = rb.sample().to(device, non_blocking=True)
        loss_vals = loss_fn(s)
        loss_vals["loss"].backward()
        optim.step()
        optim.zero_grad()
    longest = max(longest, data["step_count"].max().item())
    pbar.set_description(
        f"steps: {longest}, loss_val: {loss_vals['loss'].item(): 4.4f}, action_spread: {data['action'].sum(0)}"
    )
    exploration_module.step(data.numel())
    updater.step()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        rollout = env.rollout(10000, stoch_policy)
        traj_lens.append(rollout.get(("next", "step_count")).max().item())

######################################################################
# 결과를 도식화합니다.
#
if traj_lens:
    from matplotlib import pyplot as plt

    plt.plot(traj_lens)
    plt.xlabel("Test collection")
    plt.title("Test trajectory lengths")

######################################################################
# 결론
# ----------
#
# TorchRL에서 정책에 RNN을 통합하는 방법을 살펴보았습니다.
# 이제 다음을 수행할 수 있습니다.
#
# - :class:`~tensordict.nn.TensorDictModule` 로 작동하는 LSTM 모듈 생성하기
# - :class:`~torchrl.envs.transforms.InitTracker` 변환을 통해 LSTM 모듈에
#   초기화가 필요함을 알리기
# - 이 모듈을 정책과 손실 모듈에 통합하기
# - 수집기가 순환 상태 항목을 인식하도록 하여 나머지 데이터와 함께
#   리플레이 버퍼에 저장할 수 있도록 하기
#
# 추가 자료
# ---------------
#
# - TorchRL 문서는 `여기 <https://pytorch.org/rl/>`_ 에서 확인할 수 있습니다.
