# -*- coding: utf-8 -*-
"""
(베타) FX를 이용하여 간단한 CPU 성능 프로파일러(Profiler) 만들기
*******************************************************
**저자**: `James Reed <https://github.com/jamesr66a>`_
**번역:** `유선종 <https://github.com/Ssunbell>`_

이번 튜토리얼에서는 FX를 이용해서 다음을 진행해보겠습니다:

1) 파이토치 파이썬 코드로 코드의 구조와 실행에 대한 통계를 조사하여 수집합니다.
2) 실제 실행 단계에서 모델의 각 부분에 대해서 런타임 통계들을 수집하는 간단한 성능 "프로파일러" 역할을 할 작은 클래스를 만들어봅니다.

"""

######################################################################
# 이번 튜토리얼을 위해서, torchvision의 ResNet18 모델을 사용합니다. 데모 용도입니다.

import torch
import torch.fx
import torchvision.models as models

rn18 = models.resnet18()
rn18.eval()

######################################################################
# 이제 모델을 불러왔으므로, 모델의 성능을 좀더 깊이 조사하고자 합니다.
# 즉, 다음 호출에 대해서, 모델의 어떤 부분이 가장 오래 걸립니까?
input = torch.randn(5, 3, 224, 224)
output = rn18(input)

######################################################################
# 위의 질문에 대한 가장 일반적인 답안은 프로그램 소스를 통해 
# 프로그램의 다양한 지점에서 타임스탬프(timestamps)를 수집하는 코드를 추가하고,
# 타임스탬프 간의 차이를 비교하여 타임스탬프 간의 시간 간격을 확인하는 것입니다.
#
# 이 기술은 확실히 파이토치 코드에 적용 가능하지만 모델 코드를 복사하고 편집할 필요가 없다면,
# 특히 자신이 작성하지 않은 코드라면(이 torchvision 모델처럼) 더 좋을 것입니다. 
# FX를 사용하여 소스 코드를 수정할 필요 없이
# 이 "계측(instrumentation)" 프로세스를 자동화할 것입니다.

######################################################################
# 첫번째로, 다음의 방식처럼 몇몇 라이브러리를 불러옵니다
# (이 모든 라이브러리는 코드 뒷부분에서 사용합니다).

import statistics, tabulate, time
from typing import Any, Dict, List
from torch.fx import Interpreter

######################################################################
# .. note::
#     ``tabulate`` 는 파이토치에 종속성이 없는 외부 라이브러리입니다.
#     tabulate를 사용하여 좀더 쉽게 성능에 관한 데이터를 시각화할 것입니다.
#     여러분이 선호하는 파이썬 패키지 소스로부터 tabulate를 설치해주시길 바랍니다.

######################################################################
# 
# 상징적 추적(Symbolic Tracing)을 이용하여 모델 포착하기
# -----------------------------------------
# 다음으로, FX의 상징적 추적 메커니증을 활용하여 우리가 조작하고 
# 조사할 수 있는 자료구조에서 우리 모델의 정의를 포착할 것입니다.

traced_rn18 = torch.fx.symbolic_trace(rn18)
print(traced_rn18.graph)

######################################################################
# 이것은 ResNet18 모델의 그래프 표현을 제공합니다.
# 그래프는 서로 연결된 일련의 노드로 구성됩니다. 
# 각 노드는 Python 코드(함수, 모듈 또는 메소드 여부)에서 호출 사이트를 나타내고,
# 엣지(각 노드에서 ``args`` 및 ``kwargs`` 로 표시됨)는
# 이러한 호출 경로 사이에 전달된 값을 나타냅니다. 
# 그래프 표현과 FX의 나머지 API에 대한 자세한 정보는 
# FX 설명서 https://pytorch.org/docs/master/fx.html 에서 확인할 수 있습니다.


######################################################################
# 프로파일링 Interpreter 생성하기
# --------------------------------
# 다음으로, 우리는 ``torch.fx.Interpreter`` 로부터 상속받은 클래스를 생성할 것입니다.
# 비록 ``symbolic_trace`` 가 생성하는 ``GraphModule`` 은 ``GraphModule`` 을
# 호출할 때 실행되는 파이썬 코드를 한번에 컴파일하지만, ``GraphModule`` 을 실행하는
# 대안적인 방법은 ``graph`` 의 각 ``node`` 를 하나씩 실행하는 것입니다.
# 이것이 ``Interpreter`` 가 제공하는 기능입니다. Interpreter는 그래프를 노드 단위로 인터프리트합니다.
#
# ``Interpreter`` 로부터 상속받음으로써, 다양한 기능을 덮여씌울 수 있고 
# 원하는 프로파일링 행동을 설치할 수 있습니다. 목표는 모델을 전달하고 모델을 
# 1회 이상 호출한 다음 모델과 모델의 각 부분이 실행되는 동안 걸린 시간에 대한
# 통계를 얻을 수 있는 객체를 갖는 것입니다.
#
# ``ProfilingInterpreter`` 클래스를 정의해봅시다:

class ProfilingInterpreter(Interpreter):
    def __init__(self, mod : torch.nn.Module):
        # 사용자가 자신의 모델을 상징적으로 추적하도록 하는 것보다는,
        # 우리는 그것을 constructor에서 할 것입니다. 결과적으로
        # 사용자는 상징적 추적 API에 대한 걱정 없이 ``Module``을
        # 통과할 수 있습니다
        gm = torch.fx.symbolic_trace(mod)
        super().__init__(gm)

        # 우리는 여기에 두 가지를 저장할 것입니다:
        #
        # 1. "mod"의 총 실행 시간 목록. 즉, 인터프리터가 호출될 
        #    때마다 ``mod(...)`` 시간을 저장합니다.
        self.total_runtime_sec : List[float] = []
        # 2. 노드가 실행되는 데 걸린 시간(초) 목록에 대한 ``노드`` 의 맵입니다. 
        #    이는 (1)과 유사하지만 모델의 특정 하위 부분을 볼 수 있습니다.
        self.runtimes_sec : Dict[torch.fx.Node, List[float]] = {}

    ######################################################################
    # 다음으로, 우리의 첫 번째 매서드인 ``run()`` 을 덮여씌웁니다. ``Interpreter`` 의 ``run``
    # 매서드는 모델 실행을 위한 최상위 진입점입니다. 모델의 총 런타임을 기록할 수 있도록
    # 이 매서드를 가로채고자 할 것입니다.

    def run(self, *args) -> Any:
        # 모델을 동작하기 시작한 시점을 기록합니다.
        t_start = time.time()
        # Interpreter.run()에 다시 위임하여 모델을 실행합니다.
        return_val = super().run(*args)
        # 모델 동작이 끝난 시점을 기록합니다.
        t_end = time.time()
        # ``ProfilingInterpreter`` 에 모델 실행에 걸린 총 경과 시간을 저장합니다
        self.total_runtime_sec.append(t_end - t_start)
        return return_val

    ######################################################################
    # 이제, ``run_node`` 를 덮여씌웁니다. ``Interpreter`` 는 하나의 노드를 실행하기 위해
    # ``run_node`` 를 각각 호출합니다. 모델에서 각각의 개별 호출에 걸린 시간을 기록하고
    # 측정하기 위하여 이것을 가로챌 것입니다.

    def run_node(self, n : torch.fx.Node) -> Any:
        # op를 실행한 시작 시점을 기록합니다.
        t_start = time.time()
        # Interpreter.run_node()에 다시 위임하여 op를 실행합니다.
        return_val = super().run_node(n)
        # op가 끝난 시점을 기록합니다.
        t_end = time.time()
        # runtimes_sec 자료구조에 이 노드에 대한 항목이 없으면,
        # 빈 리스트 값을 가진 항목을 추가합니다.
        self.runtimes_sec.setdefault(n, [])
        # 이 단일 호출의 총 경과 시간을 runtimes_sec 자료구조안에 기록합니다.
        self.runtimes_sec[n].append(t_end - t_start)
        return return_val

    ######################################################################
    # 마지막으로, 우리가 수집한 데이터에 대한 멋지고 조직적인 뷰를 제공하는 
    # 매서드(``Interpreter`` 매서드를 덮여씌우지 않는 방법)을 정의할 것입니다

    def summary(self, should_sort : bool = False) -> str:
        # 각 노드에 대한 요약 정보가 담긴 리스트를 선언합니다.
        node_summaries : List[List[Any]] = []
        # 전체 네트워크의 평균 런타임을 계산합니다. 프로파일링 중에 네트워크가
        # 여러 번 호출되었을 수 있기 때문에 런타임을 요약해야 합니다.
        # 이를 위해 여러 방법 중 산술 평균을 사용합니다.
        mean_total_runtime = statistics.mean(self.total_runtime_sec)

        # 각 노드에 대해 요약 통계를 기록합니다.
        for node, runtimes in self.runtimes_sec.items():
            # 비슷하게, ``node`` 에 대한 평균 런타임을 계산합니다.
            mean_runtime = statistics.mean(runtimes)
            # 더 쉬운 이해를 돕기 위해, 전체 네트워크에 대해서
            # 각각의 노드의 퍼센트 시간 또한 계산합니다.
            pct_total = mean_runtime / mean_total_runtime * 100
            # 노드의 타입, 이름, 평균 런타임 그리고 퍼센트 런타임을 기록합니다.
            node_summaries.append(
                [node.op, str(node), mean_runtime, pct_total])

        # 성능 프로파일링을 할때 대답해야 할 가장 중요한 질문 중의 하나는 "어떤 op(들)에서 가장
        # 긴 시간이 걸렸나?"입니다. 요약 뷰에서 정렬 기능을 제공함으로써 해당 질문에 대한 답을
        # 쉽게 찾아볼 수 있습니다.
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # ``tabulate`` 라이브러리를 이용해 요약 정보들을 잘 구성된 표로 만듭니다.
        headers : List[str] = [
            'Op type', 'Op', 'Average runtime (s)', 'Pct total runtime'
        ]
        return tabulate.tabulate(node_summaries, headers=headers)

######################################################################
# .. note::
#       Python의 "time.time" 함수를 사용하여 벽시계의 타임스탬프를
#       당겨서 비교합니다. 이것은 성능을 측정하는 가장 정확한 방법은 아니며
#       1차적인 근사값만 제공합니다. 이 간단한 기법은 이 튜토리얼에서 시연할
#       목적으로만 사용합니다.

######################################################################
# ResNet18의 성능 조사하기
# -----------------------------------------
# ``ProfilingInterpreter`` 를 사용하여 ResNet18 모델의 성능 특징들을 조사할 수 있습니다.

interp = ProfilingInterpreter(rn18)
interp.run(input)
print(interp.summary(True))

######################################################################
# 꼭 호출해야 할 두가지가 있습니다.
# There are two things we should call out here:
#
# * ``MaxPool2d`` 은 가장 많은 시간이 걸립니다. 이것은 잘 알려져있는 이슈입니다:
#   https://github.com/pytorch/pytorch/issues/51393
# * BatchNorm2d 또한 상당한 시간이 걸립니다. FX 튜토리얼
#   <https://tutorials.pytorch.kr/intermediate/fx_conv_bn_fuser.html>`_.
#   의 Conv-BN Fusion에서 좀더 생각할 시간을 갖고 최적화할 수 있습니다.
#
#
# 결론
# ----------
# 보시다시피 FX를 사용하면 PyTorch 프로그램(소스 코드가 없는 프로그램도!)을 
# 기계 해석이 가능한 형식으로 쉽게 포착하여 여기에서 수행한 성능 분석과 같은
# 분석에 사용할 수 있습니다. FX는 PyTorch 프로그램과 함께 작업할 수 있는
# 흥미로운 가능성의 세계를 엽니다.
#
# 마지막으로, FX는 여전히 베타 버전이기 때문에, 여러분이 이것을 사용해보시면서
# 어떤 피드백도 기꺼이 귀기울일 것입니다. 
# 파이토치 포럼(https://discuss.pytorch.org/)이나 이슈 트래커
# (https://github.com/pytorch/pytorch/issues)를 통해 
# 여러분들이 생각하시는 어떤 피드백이라도 제보해주시길 바랍니다.