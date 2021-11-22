# -*- coding: utf-8 -*-
"""
(beta) FX를 이용한 단순 CPU 성능 프로파일러 구축하기
*******************************************************
**저자** : `James Reed <https://github.com/jamesr66a>`_
**번역:** `김태희 <https://github.com/taehui530>`_

이 튜토리얼에선, FX를 사용하여 다음을 수행할 것입니다.:

1) 파이토치 파이썬 코드의 구조 및 실행에 대한 통계를 검사하고 수집할 수 있는 방식으로 코드를 캡쳐합니다.
2) 실제 실행에서 모델의 각 부분에 대한 런타임 통계를 수집하는 간단한 성능 "프로파일러" 역할을 할 작은 클래스를 구축합니다.

"""

######################################################################
# 이 튜토리얼에서는, torchvision ResNet18 모델을 시연용으로 사용할 것입니다.

import torch
import torch.fx
import torchvision.models as models

rn18 = models.resnet18()
rn18.eval()

######################################################################
# 이제 모델이 나왔으니 성능을 좀 더 자세히 살펴보도록 하겠습니다. 
# 즉, 다음 호출을 위해 모델의 어떤 부분이 가장 오래 걸릴까요?

input = torch.randn(5, 3, 224, 224)
output = rn18(input)

######################################################################
# 이 질문에 답하는 일반적인 방법은 프로그램 소스를 살펴보고, 
# 프로그램의 다양한 지점에서 타임스탬프를 수집하는 코드를 추가하고, 
# 타임스탬프 사이의 영역이 얼마나 걸리는지 알아보기 위해 
# 타임스탬프 간의 차이를 비교하는 것입니다.
#
# 이 기술은 PyTorch 코드에 적용 가능하지만, 모델 코드를 복사하여 편집할 필요가 없다면 더 좋을 것입니다.
# (특히 이 Torchvision 모델처럼 우리가 작성하지 않은 코드일 경우).
# 대신, FX를 사용하여 소스를 수정할 필요 없이 이 "계측" 과정을 자동화할 것입니다.

######################################################################
# 먼저 몇 가지를 import 처리하겠습니다
# (나중에 코드에서 이 모든 것들을 사용할 것입니다).

import statistics, tabulate, time
from typing import Any, Dict, List
from torch.fx import Interpreter

######################################################################
# .. Note::
#       ``tabulate`` 는 PyTorch에 종속되지 않은 외부 라이브러리입니다.
#       성능 데이터를 보다 쉽게 시각화하는 데 사용할 것입니다. 
#       당신이 좋아하는 파이썬 패키지 소스에서 설치했는지 확인하세요.

######################################################################
# 심볼트레이싱(Symbolic Tracing)으로 모형 캡처하기
# -----------------------------------------
# 다음으로, FX의 심볼 추적 메커니즘을 사용하여, 
# 조작하고 조사할 수 있는 데이터 구조에서 모델의 정의를 캡처할 것입니다.

traced_rn18 = torch.fx.symbolic_trace(rn18)
print(traced_rn18.graph)

######################################################################
# 이것은 ResNet18 모델의 그래프 표현을 제공합니다.
# 그래프는 서로 연결된 일련의 노드로 구성됩니다. 
# 각 노드는 파이썬 코드의 call-site를 나타내고 (함수, 모듈 또는 메소드의 여부),
# 엣지(각 노드의 ``args`` 및 ``kwargs`` 로 표시됨)는 이러한 call-site들 간에 전달된 값을 나타냅니다.
# 그래프 표현 및 나머지 FX의 API에 대한 더 자세한 내용은 FX 문서
# https://pytorch.org/docs/master/fx.html에서 확인할 수 있습니다.


######################################################################
# Profiling Interpreter 생성
#---------------------------
# 다음으로 ``torch.fx.interpreter`` 에서 상속하는 클래스를 만들겠습니다.
# 비록 ``symbolic_trace`` 가 생산하는 ``GraphModule`` 이, 당신이 ``GraphModule`` 을 호출했을때 실행되는 파이썬 코드를 컴파일 하지만, 
# ``GraphModule`` 을 실행하는 다른 방법은 ``그래프`` 의 ``노드`` 를 하나씩 실행하는 것입니다.
# 이것이 ``Interpreter`` 가 제공하는 기능입니다. : 그것은 그래프를 노드 하나하나씩 해석합니다.
#
# ``Interpreter`` 를 계승함으로써, 다양한 기능을 재정의하고 원하는 프로파일링 동작을 설치할 수 있습니다. 
# 목표는 모델을 통과시키고, 모델을 1회 이상 호출한 다음, 해당 실행 동안 모델과 모델의 각 부분이 얼마나 걸렸는지에 대해 얻을 수 있는 오브젝트를 갖는 것입니다.
#
#``Profiling Interpreter`` 클래스를  정의해봅시다.:


class ProfilingInterpreter(Interpreter):
    def __init__(self, mod : torch.nn.Module):
        # 사용자가 상징적으로 모델을 추적하도록 하는 대신, 
        # 생성자에서 모델을 추적합니다. 
        # 따라서 사용자는 심볼 추적 API에 대해 걱정할 필요 없이 어떤 ``Module`` 이라도 패스할 수 있습니다.
        gm = torch.fx.symbolic_trace(mod)
        super().__init__(gm)

        # 이곳에 두 가지를 보관해 둡니다:
        #
        # 1. ``mod`` 의 총 실행 시간 목록입니다. 
        # 다시 말해서, 이 인터프리터가 호출되어질 때 마다 ``mod(...)`` 에 걸린 시간을 따로 보관하고 있습니다.
        self.total_runtime_sec : List[float] = []
        # 2. ``노드`` 로 부터, 노드가 실행되기위해 걸린 시간의 리스트까지의 하나의 지도입니다.( 초 단위 )
        # 이것은 (1)과 비슷해보일 수 있지만, 모델의 특정 하위 부분에 대한 것으로 볼 수 있습니다. 
        self.runtimes_sec : Dict[torch.fx.Node, List[float]] = {}

    ######################################################################
    # 다음은, 첫번째 메소드를 재정의 해봅시다. : ``run()``. 
    # ``Interpreter`` 의 ``run`` 메소드는 모델 실행을 위한 최상위 진입점입니다.
    # 모델의 총 런타임을 기록할 수 있도록 이것을 가로채야 합니다.
    

    def run(self, *args) -> Any:
        # 모델을 실행시키기 시작한 시간을 기록합니다.
        t_start = time.time()
        # Interprector.run()으로 다시 위임하여 모델을 실행합니다.
        return_val = super().run(*args)
        # 모델 실행 완료 시간을 기록합니다.
        t_end = time.time()
        # ProfilingInterpreter에 본 모델 실행의 총 경과시간을 저장합니다.
        self.total_runtime_sec.append(t_end - t_start)
        return return_val

    ######################################################################
    # 이제, ``run_node`` 를 재정의합시다.
    # ``Interpreter`` 는 단일 노드를 실행할 때마다 ``run_node`` 를 호출합니다.
    # 모델에서의 개별 호출에 걸리는 시간을 측정하고 기록할 수 있도록 이것을 가로챌 것입니다.

    def run_node(self, n : torch.fx.Node) -> Any:
        # op를 실행하기 시작한 시간을 기록합니다. 
        t_start = time.time()
        # Interpreter.run_node()에 재위임을 함으로써 op를 실행시킵니다.
        return_val = super().run_node(n)
        # op의 실행을 끝낸 시간을 기록합니다.
        t_end = time.time()
        # runtimes_sec 데이터 구조에 이 노드에 대한 항목이 없는 경우, 목록 값이 비어 있는 노드를 추가합니다.
        self.runtimes_sec.setdefault(n, [])
        # 이 단일 호출에 대한 총 경과 시간을 런타임_sec 데이터 구조에 기록합니다.
        self.runtimes_sec[n].append(t_end - t_start)
        
        return return_val

    ######################################################################
    # 마지막으로, 우리는 우리가 수집한 데이터를 잘 정리된 시각으로 볼 수 있는 
    # 메소드를 정의할 것입니다. (어떠한 ``Interpreter`` 메소드를 재정의 하지 않는 메소드)

    def summary(self, should_sort : bool = False) -> str:
        # 각 노드별 요약정보 목록을 작성합니다.
        node_summaries : List[List[Any]] = []
        # 전체 네트워크의 평균 런타임을 계산합니다. 
        # 프로파일링 중에 네트워크가 여러 번 호출되었을 수 있으므로 실행 시간을 요약해야 합니다. 
        # 이것에 대해 산술 평균을 사용해봅시다.
        mean_total_runtime = statistics.mean(self.total_runtime_sec)

        # 각 노드별 요약 통계를 기록합니다.
        for node, runtimes in self.runtimes_sec.items():
            # 이와 유사하게, ``노드`` 의 평균 런타임을 계산합니다.
            mean_runtime = statistics.mean(runtimes)
            # 보다 쉽게 이해할 수 있도록, 
            # 전체 네트워크에 관하여 각 노드가 소요된 시간의 백분율도 계산합니다.
            pct_total = mean_runtime / mean_total_runtime * 100
           # 노드의 유형, 노드 이름, 평균 런타임 및 실행률을 기록합니다.
            node_summaries.append(
                [node.op, str(node), mean_runtime, pct_total])
       
        # 성능 프로파일링을 할 때 가장 중요한 질문 중 하나는 "어떤 작업이 가장 오래 걸립니까?"입니다.
        # 요약 보기(summary view)에서 정렬 기능을 제공함으로써 이것을 쉽게 볼 수 있습니다.
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # 요약 정보를 표시하는 올바른 형식의 표를 만들려면 ``tabulate`` 라이브러리를 사용하세요.
        headers : List[str] = [
            'Op type', 'Op', 'Average runtime (s)', 'Pct total runtime'
        ]
        return tabulate.tabulate(node_summaries, headers=headers)

######################################################################
# ..Note::
#       우리는 wall clock의 타임스탬프들을 끌어와 그들을 비교하는 파이썬의 ``time.time`` 함수를 사용합니다. 
#       이 방법은 성능을 측정하는 가장 정확한 방법이 아니며, 1차 근사치만 제공합니다.
#       이 간단한 기술은 이 튜토리얼의 시연 목적으로만 사용됩니다.

######################################################################
# ResNet18의 성능 조사
# -----------------------------------------
# 이제 ResNet18 모델의 성능 특성을 점검하기 위해서 ``ProfilingInterpreter`` 을 쓸 수 있습니다.

interp = ProfilingInterpreter(rn18)
interp.run(input)
print(interp.summary(True))

######################################################################
# 여기서 호출해야할 할 두 가지가 있습니다.:
#
# * MaxPool2d가 가장 많은 시간을 차지합니다.
# 이것은 알려진 문제입니다: https://github.com/pytorch/pytorch/issues/51393
# * BatchNorm2d도 상당한 시간이 소요됩니다.
# Conv-BN Fusion에서 FX 튜토리얼 TODO 를 통해 이러한 생각을 계속하고 이것을 최적화할 수 있습니다. : 링크
# 
# 
# 결론
# ---------- 
# 보시다시피, FX를 사용하면 PyTorch 프로그램(소스 코드를 갖고있지 않은 프로그램도 포함)을 
# 기계 해석 가능한 형식으로 쉽게 캡처하여, 이곳에서 이뤄진 성능 분석과 같은 분석에 사용할 수 있습니다.
# FX는 PyTorch 프로그램으로 작업할 수 있는 흥미로운 가능성의 세계를 열어줍니다.
#
# 마지막으로, FX는 아직 베타 버전이기 때문에 FX 사용에 대한 귀하의 의견을 듣고 싶습니다.
# 당신이 가지고 있는 피드백을 제공하기 위해서 언제든지 PyTorch 포럼(https://discuss.pytorch.org/)과 문제 추적기(https://github.com/pytorch/pytorch/issues)를 사용하세요.
