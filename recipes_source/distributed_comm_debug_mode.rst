PyTorch에서 ``CommDebugMode`` 시작하기
=====================================================

**저자**: `Anshul Sinha <https://github.com/sinhaanshul>`__

이 튜토리얼에서는 PyTorch의 DistributedTensor(DTensor)와 함께 ``CommDebugMode``를 사용하는 방법을 살펴봅니다.  
이를 통해 분산 학습 환경에서 수행되는 집합 연산(collective operation)을 추적하여 디버깅할 수 있습니다.

사전 준비(Prerequisites)
---------------------

* Python 3.8 - 3.11
* PyTorch 2.2 이상


``CommDebugMode``란 무엇이며, 왜 유용한가
----------------------------------------------------
모델의 크기가 커짐에 따라, 사용자는 다양한 병렬화(parallelism) 전략을 조합하여 분산 학습(distributed training)을 확장하려 합니다.  
하지만 기존 솔루션 간의 상호운용성(interoperability) 부족은 여전히 큰 과제로 남아 있습니다.  
이는 서로 다른 병렬화 전략을 연결할 수 있는 통합된 추상화(unified abstraction)가 부족하기 때문입니다.

이 문제를 해결하기 위해 PyTorch는 `DistributedTensor(DTensor)  
<https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/examples/comm_mode_features_example.py>`_ 를 도입했습니다.  
DTensor는 분산 학습 환경에서 텐서 통신의 복잡성을 추상화하여 사용자에게 일관되고 간결한 경험을 제공합니다.  

그러나 이러한 통합 추상화를 사용하는 과정에서, 내부적으로 어떤 시점에 집합 통신이 수행되는지 명확히 알기 어려워  
고급 사용자가 디버깅하거나 문제를 식별하기 어렵습니다.  

이때 ``CommDebugMode``는 Python의 컨텍스트 매니저(context manager)로서  
DTensor 사용 중 발생하는 집합 연산의 시점과 이유를 시각적으로 추적할 수 있는 주요 디버깅 도구입니다.  
이를 통해 사용자는 언제, 왜 collective 연산이 실행되는지를 명확히 파악할 수 있습니다.


``CommDebugMode`` 사용법
------------------------

다음은 ``CommDebugMode``를 사용하는 예시입니다:

.. code-block:: python

    # 이 예제에서 사용된 모델은 텐서 병렬화(tensor parallelism)를 적용한 MLPModule입니다.
    comm_mode = CommDebugMode()
    with comm_mode:
        output = model(inp)

    # 연산 단위의 collective 추적 정보를 출력
    print(comm_mode.generate_comm_debug_tracing_table(noise_level=0))

    # 연산 단위의 collective 추적 정보를 파일로 기록
    comm_mode.log_comm_debug_tracing_table_to_file(
        noise_level=1, file_name="transformer_operation_log.txt"
    )

    # 연산 단위의 collective 추적 정보를 JSON 파일로 덤프(dump)
    # 아래의 시각화 브라우저에서 이 JSON 파일을 사용할 수 있습니다.
    comm_mode.generate_json_dump(noise_level=2)


다음은 noise level 0에서 MLPModule의 출력 예시입니다:

.. code-block:: python

    Expected Output:
        Global
          FORWARD PASS
            *c10d_functional.all_reduce: 1
            MLPModule
              FORWARD PASS
                *c10d_functional.all_reduce: 1
                MLPModule.net1
                MLPModule.relu
                MLPModule.net2
                  FORWARD PASS
                    *c10d_functional.all_reduce: 1


``CommDebugMode``를 사용하려면 모델 실행 코드를 ``CommDebugMode`` 블록 안에 감싸고,  
원하는 정보를 표시하는 API를 호출하면 됩니다.  

또한 ``noise_level`` 인자를 사용해 출력되는 정보의 상세 수준(verbosity level)을 제어할 수 있습니다.  
각 noise level은 다음과 같은 정보를 제공합니다:

| 0. 모듈 단위의 collective 연산 개수 출력  
| 1. 중요하지 않은 연산을 제외한 DTensor 연산 및 모듈 샤딩(sharding) 정보 출력  
| 2. 중요하지 않은 연산을 제외한 텐서 단위 연산 출력  
| 3. 모든 연산 출력  

위의 예시에서 볼 수 있듯이, collective 연산인 all_reduce는 ``MLPModule``의 forward 단계에서 한 번 발생합니다.  
또한 ``CommDebugMode``를 사용하면 이 all-reduce 연산이 ``MLPModule``의 두 번째 선형 계층(linear layer)에서  
발생한다는 점을 정확히 확인할 수 있습니다.


아래는 생성된 JSON 파일을 업로드하여 시각적으로 탐색할 수 있는  
인터랙티브 모듈 트리 시각화(interactive module tree visualization)입니다:

.. raw:: html

    <!DOCTYPE html>
    <html lang ="en">
    <head>
        <meta charset="UTF-8">
        <meta name = "viewport" content="width=device-width, initial-scale=1.0">
        <title>CommDebugMode Module Tree</title>
        <style>
            ul, #tree-container {
                list-style-type: none;
                margin: 0;
                padding: 0;
            }
            .caret {
                cursor: pointer;
                user-select: none;
            }
            .caret::before {
                content: "\25B6";
                color:black;
                display: inline-block;
                margin-right: 6px;
            }
            .caret-down::before {
                transform: rotate(90deg);
            }
            .tree {
                padding-left: 20px;
            }
            .tree ul {
                padding-left: 20px;
            }
            .nested {
                display: none;
            }
            .active {
                display: block;
            }
            .forward-pass,
            .backward-pass {
                margin-left: 40px;
            }
            .forward-pass table {
                margin-left: 40px;
                width: auto;
            }
            .forward-pass table td, .forward-pass table th {
                padding: 8px;
            }
            .forward-pass ul {
                display: none;
            }
            table {
                font-family: arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            td, th {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            tr:nth-child(even) {
                background-color: #dddddd;
            }
            #drop-area {
                position: relative;
                width: 25%;
                height: 100px;
                border: 2px dashed #ccc;
                border-radius: 5px;
                padding: 0px;
                text-align: center;
            }
            .drag-drop-block {
                display: inline-block;
                width: 200px;
                height: 50px;
                background-color: #f7f7f7;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                color: #666;
                cursor: pointer;
            }
            #file-input {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
            }
        </style>
    </head>
    <body>
        <div id="drop-area">
            <div class="drag-drop-block">
              <span>Drag file here</span>
            </div>
            <input type="file" id="file-input" accept=".json">
          </div>
        <div id="tree-container"></div>
        <script src="https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/distributed/tensor/debug/comm_mode_broswer_visual.js"></script>
    </body>
    </html>


결론(Conclusion)
------------------------------------------

이 레시피에서는 PyTorch의 ``CommDebugMode``를 사용하여  
집합 통신(collective communication)을 포함하는 DistributedTensor 및 병렬화 솔루션을 디버깅하는 방법을 배웠습니다.  
또한 생성된 JSON 출력을 내장된 시각화 브라우저에서 직접 불러와 확인할 수도 있습니다.  

``CommDebugMode``에 대한 보다 자세한 내용은  
`comm_mode_features_example.py  
<https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/examples/comm_mode_features_example.py>`_ 를 참고하세요.