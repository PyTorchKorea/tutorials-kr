
비동기 실행을 사용하여 배치 RPC 프로세싱 구현하기
===============================================================
**저자**: `Shen Li <https://mrshenli.github.io/>`_
**번역**: 'RushBsite <https://github.com/RushBsite>

선수과목(Prerequisites):

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `Getting started with Distributed RPC Framework <rpc_tutorial.html>`__
-  `Implementing a Parameter Server using Distributed RPC Framework <rpc_param_server_tutorial.html>`__
-  `RPC Asynchronous Execution Decorator <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__


이 튜토리얼에서는 `@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__ 데코레이터(decorator)를 사용하여
배치-프로세싱(batch-processing) 원격 프로시져 호출(RPC, Remote Procedure Call) 어플리케이션을 빌드하는 방법을 설명합니다. 이 방법은 차단된 RPC 스레드 수를 줄이고 
수신자(callee)의 CUDA 작업을 통합하여 학습 속도를 높이는 데 도움이 됩니다.
이 아이디어는 `Batch Inference with TorchServer <https://pytorch.org/serve/batch_inference_with_ts.html>`__ 와 동일한 아이디어를 공유합니다.


.. note:: 이 튜토리얼은 PyTorch v1.6.0 이상을 필요로 합니다.

기초
------

이전 튜토리얼들에서는 `torch.distributed.rpc <https://pytorch.org/docs/stable/rpc.html>`__ 를 사용하여 분산 학습을 구축하는 단계를 확인했습니다.
하지만, RPC 리퀘스트가 처리될 때 수신자 측에서 일어나는 일은 자세히 설명하지 않았습니다. PyTorch v1.5부터 각 RPC 리퀘스트는 호출한 수신자에게
해당 함수가 반환될 때까지 해당 리퀘스트에서 함수를 실행합니다. 이는 많은 사례에 적용되지만, 한 가지 주의사항이 있습니다. 사용자의 함수에서
IO의 블록 (예 : 중첩된 RPC 호출 또는 시그널링(signaling) 포함) 차단 해제를위한 다른 RPC 요청이 있는 경우 수신자의 RPC 스레드는 IO가 완료되거나 신호 이벤트가 발생할 때까지 유휴
대기합니다. 따라서, RPC 수신자가 필요 이상으로 많은 스레드를 사용하게 됩니다. 문제의 원인은 RPC가 사용자 기능을 일종의 블랙박스로 취급하고 함수에서 일어나는 일에 무지한
것에 있습니다. 사용자 함수에 원활한 잉여 RPC 스레드를 할당하기 위해선, RPC 시스템에 더 많은 정보를 제공해야 합니다.

v1.6.0 버전부터 PyTorch 는 두 가지 새로운 개념을 도입하여 이 문제를 해결합니다 :


* `torch.futures.Future <https://pytorch.org/docs/master/futures.html>`__ 로 비동기 실행을 캡슐화하며, 콜백 함수 설치를 지원합니다.

* `@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__
  데코레이터는 어플리케이션이 수신자에게 대상 함수의 반환 여부를 전달하게 하고 실행 중에 여러번 일시 중지, 양보를 가능하게 합니다.


이 두 도구를 사용하면, 응용 프로그램 코드가 사용자 함수를 여러 개의 작은 함수들로 분할하여, ``Future`` 오브젝트의 콜백 함수로 연결하고
최종 결과를 포함하는 ``Future`` 오브젝트를 반환합니다. 수신자 측은 ``Future`` 오브젝트를 수신할 때, 최종 결과가 준비된 상태라면,
콜백으로서 준비 및 통신을 후속 RPC 응답으로써 설치합니다. 이 경우, 수신 측에서는 최종 결괏값이 반환될 때까지 기다리며 스레드를 
차단할 필요가 없습니다. 간단한 예제는 다음 API 문서를 참조하세요.`@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__


수신자의 유휴 스레드 수를 줄이는 것 외에도 이러한 도구는 일괄 RPC 처리를 더 쉽고 빠르게 할 수 있습니다. 
다음 두 튜토리얼 섹션은 분산된 배치 업데이트 파라미터 서버를 구축하는 방법과
`@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__
데코레이터를 사용한 일괄 처리 강화 학습 애플리케이션 을 빌드하는 방법을 보여줍니다.


배치 업데이트 파라미터 서버
-------------------------------

동기화된 파라미터 서버(parameter server) 학습 어플리케이션을 하나의 파라미터 서버(PS) 와 다중 트레이너로 사용하는 경우를 생각해봅시다.
이 어플리케이션에선, PS는 매개변수를 유지하고 모든 트레이너가 변화도(gradients)를 보고할 때 까지 기다립니다. 매 시퀀스마다
모든 트레이너에게 변화도를 수신할 때까지 기다린 후, 한 번에 모든 매개변수를 업데이트합니다. 아래에 있는 코드는
PS 클래스의 구현을 보여줍니다. ``update_and_fetch_model`` 방법은 ``@rpc.functions.async_execution`` 을 사용하여 데코레이터에
이용되고, 트레이너로부터 호출됩니다. 각 호출에선 업데이트된 모델로 채워질 ``Future`` 오브젝트를 반환합니다. 대부분의 트레이너로부터
발생한 호출은 축적된 변화도를 ``.grad`` 필드로 즉시 반환하고 PS에서 RPC 스레드를 생성합니다. 마지막 트레이너는 최적화 단계를
트리거하고 이전에 기록된 모든 변화도를 소비합니다. 그다음, 업데이트된 모델로 ``future_model`` 을 설정합니다. 이 모델은
다른 트레이너의 모든 이전 요청을 ``Future`` 객체를 통해 차례로 통지하고 업데이트된 모델을 모든 트레이너에게 보냅니다.


.. code:: python

    import threading
    import torchvision
    import torch
    import torch.distributed.rpc as rpc
    from torch import optim

    num_classes, batch_update_size = 30, 5

    class BatchUpdateParameterServer(object):
        def __init__(self, batch_update_size=batch_update_size):
            self.model = torchvision.models.resnet50(num_classes=num_classes)
            self.lock = threading.Lock()
            self.future_model = torch.futures.Future()
            self.batch_update_size = batch_update_size
            self.curr_update_size = 0
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            for p in self.model.parameters():
                p.grad = torch.zeros_like(p)

        def get_model(self):
            return self.model

        @staticmethod
        @rpc.functions.async_execution
        def update_and_fetch_model(ps_rref, grads):
            # 로컬 PS 인스턴스를 검색하기 위해 RRef 사용
            self = ps_rref.local_value()
            with self.lock:
                self.curr_update_size += 1
                # 변화도를 .grad 필드에 저장(축적)
                for p, g in zip(self.model.parameters(), grads):
                    p.grad += g

                # 이 스레드가 반환되기 전에 다른 퓨처 오브젝트(future object) 가 올바른 모델을 홀딩 하는지 검수하기 위해
                # 현재 future_model 을 저장하고 반환
                fut = self.future_model

                if self.curr_update_size >= self.batch_update_size:
                    # 모델 업데이트
                    for p in self.model.parameters():
                        p.grad /= self.batch_update_size
                    self.curr_update_size = 0
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # 퓨처 오브젝트의 결과값을 설정함으로써, 이 모델을 업데이트 하기 위한
                    # 모든 이전 리퀘스트에게 결과값을 전달
                    fut.set_result(self.model)
                    self.future_model = torch.futures.Future()

            return fut


트레이너들의 경우, PS의 동일한 매개변수 세트를 사용하여 초기화됩니다. 매 시퀀스마다 각 트레이너들은 먼저
변화도를 로컬하게 생성하기 위해 포워드, 백워드 패스를 실행합니다. 그리고 각 트레이너들은 RPC를 사용하여 PS에서
변화도를 보고하고 동일한 RPC 리퀘스트의 반환값을 통해 업데이트된 매개변수를 반환받습니다. 트레이너의 구현에선
대상함수가 ``@rpc.functions.async_execution`` 로 마크 되는지 여부는 결과에 차이가 없습니다. 트레이너는 단순히
``rpc_sync`` 를 사용하여 ``update_and_fetch_model`` 을 호출하고 이 모델은 업데이트 된 모델이 반환 될 때까지 트레이너에서 차단합니다.

.. code:: python

    batch_size, image_w, image_h  = 20, 64, 64

    class Trainer(object):
        def __init__(self, ps_rref):
            self.ps_rref, self.loss_fn = ps_rref, torch.nn.MSELoss()
            self.one_hot_indices = torch.LongTensor(batch_size) \
                                        .random_(0, num_classes) \
                                        .view(batch_size, 1)

        def get_next_batch(self):
            for _ in range(6):
                inputs = torch.randn(batch_size, 3, image_w, image_h)
                labels = torch.zeros(batch_size, num_classes) \
                            .scatter_(1, self.one_hot_indices, 1)
                yield inputs.cuda(), labels.cuda()

        def train(self):
            name = rpc.get_worker_info().name
            # model 매개변수 초기값 설정
            m = self.ps_rref.rpc_sync().get_model().cuda()
            # 트레이닝 시작
            for inputs, labels in self.get_next_batch():
                self.loss_fn(m(inputs), labels).backward()
                m = rpc.rpc_sync(
                    self.ps_rref.owner(),
                    BatchUpdateParameterServer.update_and_fetch_model,
                    args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
                ).cuda()



이 튜토리얼에서는 멀티 프로세스를 실행하는 코드를 생략합니다. 코드 전문은 `examples <https://github.com/pytorch/examples/tree/master/distributed/rpc>`__
저장소를 참조하십시오. `@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__
데코레이터 없이도 배치 프로세싱 을 구현하는 것이 가능하지만 PS에서 더 많은 RPC 스레드를 블록하거나 더 많은 RPC 시퀀스를 모델 업데이트에 소비해야하고 이는
코드의 복잡성과 통신에서 오버헤드 발생을 증가시킵니다.

이 섹션에선 간단한 파라미터 서버 학습 예제를 이용하여 `@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__
를 사용하는 batch RPC 어플리케이션의 구현 방법을 설명합니다. 다음 섹션에선 이전 강화 학습 예제 튜토리얼 `Getting started with Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`__
을 배치 프로세싱 으로 재 구현하고 학습 속도에 미치는 영향을 알아봅니다.

배치 프로세싱을 활용한 카트-폴 해결(CartPole Solver)
--------------------------------

이 섹션에서는 `OpenAI Gym <https://gym.openai.com/>`__ 의 CartPole-v1을 배치 프로세싱 RPC 의 활용 효과를 보여주기 위한 예시로써 사용합니다.
최적의 카트폴 알고리즘이나 상극의 RL 문제를 해결하는것이 목적이 아니라, `@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__
의 활용을 확인하는것 이 목적임을 유의하시기 바랍니다. 따라서 매우 간단한 정책과 보상 계산 전략을 사용하고 다중 관찰자 단일 에이전트(agent) 배치 RPC 구현에 중점을 둡니다.
우리는 아래에 표시된 이전 튜토리얼과 유사한 ``Policy`` 모델을 사용할 것입니다. 이전 튜토리얼과 비교했을때, 생성자가 ``F.softmax`` 를 위한 ``dim`` 매개변수를 제어하는 추가적인
 ``batch`` 인수를 배칭을 위해 생성하고, ``forward`` 함수의 ``x`` 인자는 여러 관찰자의 상태를 포함하므로 적절한 차수 변화가 필요합니다. 다른 모든 것은 그대로 유지됩니다.

.. code:: python

    import argparse
    import torch.nn as nn
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description='PyTorch RPC Batch RL example')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                        help='discount factor (default: 1.0)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--num-episode', type=int, default=10, metavar='E',
                        help='number of episodes (default: 10)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    class Policy(nn.Module):
        def __init__(self, batch=True):
            super(Policy, self).__init__()
            self.affine1 = nn.Linear(4, 128)
            self.dropout = nn.Dropout(p=0.6)
            self.affine2 = nn.Linear(128, 2)
            self.dim = 2 if batch else 1

        def forward(self, x):
            x = self.affine1(x)
            x = self.dropout(x)
            x = F.relu(x)
            action_scores = self.affine2(x)
            return F.softmax(action_scores, dim=self.dim)




``Observer`` 의 생성자도 역시 적절하게 조정해야합니다. 여기에서도 역시 ``Agent`` 함수에서 선택 액션에 사용되는 ``batch`` 인수를 가집니다.
배치 모드에서는 곧 소개할 ``Agent`` 에서 ``select_action_batch`` 함수를 호출합니다. 이 함수는 `@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__.
에 의해 데코레이트 됩니다.

.. code:: python

    import gym
    import torch.distributed.rpc as rpc

    class Observer:
        def __init__(self, batch=True):
            self.id = rpc.get_worker_info().id - 1
            self.env = gym.make('CartPole-v1')
            self.env.seed(args.seed)
            self.select_action = Agent.select_action_batch if batch else Agent.select_action



이전 튜토리얼 `Getting started with Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`__ 과 비교했을때
관측자의 구성이 약간 달라졌습니다. 환경이 정지되었을때 종료하는 대신, 모든 에피소드(episode)에서 항상 ``n_steps`` 반복을 실행합니다. 환경의 상태가 
돌아오면, 관찰자는 단순히 환경을 재설정하고 다시 시작합니다. 이 디자인을 사용하면 에이전트는 모든 관찰자를 고정 된 크기의 텐서로 압축 할 수 있기 때문에
고정된 수의 상태를 수신합니다. 매 단계에서, ``Observer`` 는 RPC를 사용하여 ``Agent`` 에 상태를 보내고 반환 값을 통한 액션을 가져옵니다. 매 에피소드가
종료될 때 마다 모든 단계의 보상을 ``Agent`` 에게 반환합니다. 이 ``run_episode`` 함수는 RPC를 사용하여 ``Agent`` 를 호출하는것에 유의하십시오.
따라서이 함수의 ``rpc_sync`` 호출은 중첩 된 RPC 호출이 됩니다. 
또한 ``Observer`` 에서 하나의 스레드를 차단하지 않도록 이 함수를 ``@ rpc.functions.async_execution`` 으로 표시 할 수 있습니다. 그러나
``Observer`` 대신 ``Agent`` 의 병목 현상으로 인해,  ``Observer`` 프로세스의 스레드를 차단하는 것도 고려해 볼 수 있습니다.

.. code:: python

    import torch

    class Observer:
        ...

        def run_episode(self, agent_rref, n_steps):
            state, ep_reward = self.env.reset(), NUM_STEPS
            rewards = torch.zeros(n_steps)
            start_step = 0
            for step in range(n_steps):
                state = torch.from_numpy(state).float().unsqueeze(0)
                # agent에게 현재 state 전달하여 action 실행
                action = rpc.rpc_sync(
                    agent_rref.owner(),
                    self.select_action,
                    args=(agent_rref, self.id, state)
                )

                # environment 에게 action 전달하고 reward를 get
                state, reward, done, _ = self.env.step(action)
                rewards[step] = reward

                if done or step + 1 >= n_steps:
                    curr_rewards = rewards[start_step:(step + 1)]
                    R = 0
                    for i in range(curr_rewards.numel() -1, -1, -1):
                        R = curr_rewards[i] + args.gamma * R
                        curr_rewards[i] = R
                    state = self.env.reset()
                    if start_step == 0:
                        ep_reward = min(ep_reward, step - start_step + 1)
                    start_step = step + 1

            return [rewards, ep_reward]



``Agent`` 의 생성자 역시 ``batch`` 인자를 가집니다. 이 인자는 액션 프롭(action probs)이 어떻게 
배치 프로세싱 되는지 제어합니다. 배치 모드에서 ``saved_log_probs`` 에는
한 단계의 모든 관측자의 액션 프롭이 포함되어있는 텐서의 리스트를 포함합니다. 배치 프로세싱이 존재
하지 않으면, ``saved_log_probs`` 는 관찰자 ID를 키값으로 가지고 관측자의 액션 프롭에 대한 리스트를
밸류 값으로 가지는 딕셔너리 자료형(dictionary) 입니다.


.. code:: python

    import threading
    from torch.distributed.rpc import RRef

    class Agent:
        def __init__(self, world_size, batch=True):
            self.ob_rrefs = []
            self.agent_rref = RRef(self)
            self.rewards = {}
            self.policy = Policy(batch).cuda()
            self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
            self.running_reward = 0

            for ob_rank in range(1, world_size):
                ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
                self.ob_rrefs.append(rpc.remote(ob_info, Observer, args=(batch,)))
                self.rewards[ob_info.id] = []

            self.states = torch.zeros(len(self.ob_rrefs), 1, 4)
            self.batch = batch
            self.saved_log_probs = [] if batch else {k:[] for k in range(len(self.ob_rrefs))}
            self.future_actions = torch.futures.Future()
            self.lock = threading.Lock()
            self.pending_states = len(self.ob_rrefs)


배치 프로세싱이 아닌 ``select_acion`` 은 단순히 상태를 실행하여 정책을 실행하고 저장합니다.
액션 프롭을 확인하고 즉시 관찰자에게 액션을 반환합니다.

.. code:: python

    from torch.distributions import Categorical

    class Agent:
        ...

        @staticmethod
        def select_action(agent_rref, ob_id, state):
            self = agent_rref.local_value()
            probs = self.policy(state.cuda())
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs[ob_id].append(m.log_prob(action))
            return action.item()



배치 프로세싱을 활용하면 2차원 텐서에 저장된 상태인 ``self.states`` 는 관찰자 id를 행간 id로써 사용합니다.
그리고 ``Futre`` 오브젝트를 배치-생성된 콜백함수인  ``self.future_actions`` ``Future`` 에 연결함으로써,
해당 관찰자의 ID를 사용하여 인덱싱 된 특정 행으로 채워집니다. 마지막으로 도착한 관찰자는 한번에 모든 
배치 상태를 실행하고 그에 따라  ``self.future_actions`` 를 설정합니다. 이 경우 모든 ``self.future_actions`` 에 설치된 콜백 함수가 트리거되고
반환 값은 연결된 ``Future`` 오브젝트를 를 채우는 데 사용됩니다. 이는 차례로 ``Agent`` 에게 다른 관찰자의 모든 이전 RPC 요청의 응답을 준비하고 전달하도록 알립니다.

.. code:: python

    class Agent:
        ...

        @staticmethod
        @rpc.functions.async_execution
        def select_action_batch(agent_rref, ob_id, state):
            self = agent_rref.local_value()
            self.states[ob_id].copy_(state)
            future_action = self.future_actions.then(
                lambda future_actions: future_actions.wait()[ob_id].item()
            )

            with self.lock:
                self.pending_states -= 1
                if self.pending_states == 0:
                    self.pending_states = len(self.ob_rrefs)
                    probs = self.policy(self.states.cuda())
                    m = Categorical(probs)
                    actions = m.sample()
                    self.saved_log_probs.append(m.log_prob(actions).t()[0])
                    future_actions = self.future_actions
                    self.future_actions = torch.futures.Future()
                    future_actions.set_result(actions.cpu())
            return future_action



이제 서로 다른 RPC 기능이 함께 연결되는 방식을 정의하겠습니다. ``Agent`` 는
모든 에피소드의 실행을 제어합니다. ``Agent`` 는 먼저 ``rpc_async`` 를 사용하여 
모든관측자에 대한 에피소드 및 관찰자 보상으로 이루어진 반환된 futures 를 차단합니다.
아래 코드는 ``ob_rref.rpc_async()`` RRef helper를 사용하여 ``ob_rref`` RRef의 제공된 인수로 ``run_episode`` 함수를 실행 합니다.
그 다음, 저장된 액션 프롭과 반환된 관찰자 보상을 기반으로한 예상 데이터 형식을 선택하고, 훈련 단계를 시작합니다.
마지막으로 모든 항목을 재설정하고 현재 에피소드의 보상을 표시하고 반환합니다. 이 함수는 하나의 에피소드를 실행하는 시작점이 됩니다.

.. code:: python

    class Agent:
        ...

        def run_episode(self, n_steps=0):
            futs = []
            for ob_rref in self.ob_rrefs:
                # async RPC가 다른 관측자(observer)들을 차단하게 함
                futs.append(ob_rref.rpc_async().run_episode(self.agent_rref, n_steps))

            # 이 에피소드 가 끝날때까지 모든 관찰자 대기
            rets = torch.futures.wait_all(futs)
            rewards = torch.stack([ret[0] for ret in rets]).cuda().t()
            ep_rewards = sum([ret[1] for ret in rets]) / len(rets)

            # stack 은 prob 를 tensor로 저장
            if self.batch:
                probs = torch.stack(self.saved_log_probs)
            else:
                probs = [torch.stack(self.saved_log_probs[i]) for i in range(len(rets))]
                probs = torch.stack(probs)

            policy_loss = -probs * rewards / len(rets)
            policy_loss.sum().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 변수 재설정
            self.saved_log_probs = [] if self.batch else {k:[] for k in range(len(self.ob_rrefs))}
            self.states = torch.zeros(len(self.ob_rrefs), 1, 4)

            # running reward 계산
            self.running_reward = 0.5 * ep_rewards + 0.5 * self.running_reward
            return ep_rewards, self.running_reward



코드의 나머지 부분은 다른 RPC 튜토리얼의 실행과 로깅(logging)에 관한 일반적인 프로세싱과 유사합니다.
이 튜토리얼에선 모든 관측자들은 수동적으로 에이전트의 명령을 기다립니다. 자세한 예시와 구성은 `examples <https://github.com/pytorch/examples/tree/master/distributed/rpc>`__
저장소를 참고하십시오.

.. code:: python

    def run_worker(rank, world_size, n_episode, batch, print_log=True):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        if rank == 0:
            # rank0 은 agent
            rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

            agent = Agent(world_size, batch)
            for i_episode in range(n_episode):
                last_reward, running_reward = agent.run_episode(n_steps=NUM_STEPS)

                if print_log:
                    print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                        i_episode, last_reward, running_reward))
        else:
            # 다른 rank 들은 관측자
            rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
            # 관측자들은 수동적으로 에이전트의 지시를 기다림
        rpc.shutdown()


    def main():
        for world_size in range(2, 12):
            delays = []
            for batch in [True, False]:
                tik = time.time()
                mp.spawn(
                    run_worker,
                    args=(world_size, args.num_episode, batch),
                    nprocs=world_size,
                    join=True
                )
                tok = time.time()
                delays.append(tok - tik)

            print(f"{world_size}, {delays[0]}, {delays[1]}")


    if __name__ == '__main__':
        main()



배치 RPC는 작업 추론을 적은 CUDA 작업으로 통합하는데 도움이 될 뿐만 아니라 오버헤드의 전달또한 
감소시킵니다. 상단의 ``main`` 함수는 배치된 상태와 배치되지 않은 상태 의 두 모드를 1과 10사이의 범위의 서로 다른 수의 관측자를 
사용하여 실행시킵니다. 아래 그림은 기본 인수 값을 사용하는 서로 다른 구성 크기의 실행 시간을 나타냅니다. 이
결과로써 배치 프로세싱이 학습 속도를 향상시키는데 도움외 된다는 것을 확인할 수 있습니다.


.. figure:: /_static/img/rpc-images/batch.png
    :alt:

더 알아보기
----------

-  `Batch-Updating Parameter Server Source Code <https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py>`__
-  `Batch-Processing CartPole Solver <https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/reinforce.py>`__
-  `Distributed Autograd <https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework>`__
-  `Distributed Pipeline Parallelism <dist_pipeline_parallel_tutorial.html>`__