PyTorch RPC 기반 워크로드 프로파일링
======================================

이 레시피에서는 다음을 학습합니다.

-  `Distributed RPC Framework`_ 개요
-  `PyTorch Profiler`_ 개요
-  프로파일러를 사용하여 RPC 기반 워크로드를 프로파일링 하는 방법

선행학습
------------

-  PyTorch 1.6

PyTorch의 설치를 위한 지침은 `pytorch.org`_ 에서 확인하실 수 있습니다.

분산(Distributed) RPC Framework란?
---------------------------------------

**분산 RPC Framework**\는 원격 통신을 허용하는 primitives 집합을 통해 다중 머신 모델 학습을 위한 메커니즘과, 여러 머신에 분할된 모델을 자동으로 구별하는 상위 레벨의 API를 제공합니다. 이 레시피에서는 `Distributed RPC Framework`_ 와 `RPC Tutorials`_ 에 익숙해지는 것이 도움이 됩니다. 


PyTorch 프로파일러(Profiler)란?
---------------------------------------
프로파일러는 모델 워크로드에서 연산자(operators)의 수요에 따른(on-demand) 프로파일링을 허용하는 API기반의 컨텍스트 관리자입니다. 프로파일러는 실행시간, 호출된 연산자와 메모리 소비를 포함하여 모델의 다양한 측면을 분석하는데 사용될 수 있습니다. 단일 노드 모델을 프로파일링 하기 위해 프로파일러를 사용하는 자세한 자습서는 `Profiler Recipe`_ 를 살펴보시면 됩니다. 


RPC 기반 워크로드에 프로파일러를 사용하는 방법
-----------------------------------------------

프로파일러는 RPC로 이루어진 호출의 프로파일링을 지원하고 사용자가 다른 노드에서 발생하는 연산을 자세히 볼 수 있도록 합니다. 이에 대한 예시를 보여주기 위해 먼저 RPC 프레임워크를 설정합니다. 아래 코드 조각은 동일한 호스트에서 각각 ``worker0``\와 ``worker1``\이라고 불리는 두개의 RPC worker를 초기화합니다. worker는 하위 프로세서로서 생성되며, 우리는 적절한 초기화에 요구되는 몇가지의 환경 변수를 설정합니다.

::

  import torch
  import torch.distributed.rpc as rpc
  import torch.autograd.profiler as profiler
  import torch.multiprocessing as mp
  import os
  import logging
  import sys

  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  logger = logging.getLogger()

  def random_tensor():
      return torch.rand((3, 3), requires_grad=True)


  def worker(rank, world_size):
      os.environ["MASTER_ADDR"] = "localhost"
      os.environ["MASTER_PORT"] = "29500"
      worker_name = f"worker{rank}"

      # RPC framework 초기화.
      rpc.init_rpc(
          name=worker_name,
          rank=rank,
          world_size=world_size
      )
      logger.debug(f"{worker_name} successfully initialized RPC.")

      # 아래에서 더 추가합니다.

      logger.debug(f"Rank {rank} waiting for workers and shutting down RPC")
      rpc.shutdown()
      logger.debug(f"Rank {rank} shutdown RPC")


  if __name__ == '__main__':
      # 2개의 RPC 작업자 실행.
      world_size = 2
      mp.spawn(worker, args=(world_size,), nprocs=world_size)

위 프로그램을 실행하면 다음과 같은 출력을 얻습니다.

::

  DEBUG:root:worker1 successfully initialized RPC.
  DEBUG:root:worker0 successfully initialized RPC.
  DEBUG:root:Rank 0 waiting for workers and shutting down RPC
  DEBUG:root:Rank 1 waiting for workers and shutting down RPC
  DEBUG:root:Rank 1 shutdown RPC
  DEBUG:root:Rank 0 shutdown RPC

이제 RPC 프레임워크의 뼈대 설정을 하였으므로, RPC를 앞뒤로 보내고 프로파일러를 사용하여 내부에서 무슨일이 일어나는지 확인할 수 있습니다. 위 ``worker`` 함수에 다음을 추가합니다.

::

    def worker(rank, world_size):
        # 위 코드 생략
        if rank == 0:
            dst_worker_rank = (rank + 1) % world_size
            dst_worker_name = f"worker{dst_worker_rank}"
            t1, t2 = random_tensor(), random_tensor()
            # 프로파일링 범위하에서 전송하고 RPC 완료를 기다립니다
            with profiler.profile() as prof:
                fut1 = rpc.rpc_async(dst_worker_name, torch.add, args=(t1, t2))
                fut2 = rpc.rpc_async(dst_worker_name, torch.mul, args=(t1, t2))
                # RPC는 프로파일링 범위안에서 대기해야 합니다
                fut1.wait()
                fut2.wait()

            print(prof.key_averages().table())

앞의 코드는 각각 ``torch.add``\와 ``torch.mul``\를 구체화하는 2개의 RPC를 생성합니다. 이것들은 worker1에서 두개의 랜덤 입력 텐서로 실행됩니다. ``rpc_async`` API를 사용하기 때문에 계산 결과를 기다려야 하는 ``torch.futures.Future``\객체가 반환됩니다. 이 대기는 RPC를 정확히 프로파일링하기 위해 프로파일링 컨텍스트 관리자가 만든 범위 안에서 이루어져야 합니다. 이 새로운 작업자 함수로 코드를 실행하면 다음과 같은 결과를 얻습니다. 

::

  # 간단히 표현하기 위해 일부 열(columns)이 생략되었으며, 정확한 출력은 무작위성(randomness)을 필요로 합니다 
  ----------------------------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
  Name                                                              Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls  Node ID
  ----------------------------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
  rpc_async#aten::add(worker0 -> worker1)                           0.00%            0.000us          0                20.462ms         20.462ms         1                0
  rpc_async#aten::mul(worker0 -> worker1)                           0.00%            0.000us          0                5.712ms          5.712ms          1                0
  rpc_async#aten::mul(worker0 -> worker1)#remote_op: mul            1.84%            206.864us        2.69%            302.162us        151.081us        2                1
  rpc_async#aten::add(worker0 -> worker1)#remote_op: add            1.41%            158.501us        1.57%            176.924us        176.924us        1                1
  rpc_async#aten::mul(worker0 -> worker1)#remote_op: output_nr      0.04%            4.980us          0.04%            4.980us          2.490us          2                1
  rpc_async#aten::mul(worker0 -> worker1)#remote_op: is_leaf        0.07%            7.806us          0.07%            7.806us          1.952us          4                1
  rpc_async#aten::add(worker0 -> worker1)#remote_op: empty          0.16%            18.423us         0.16%            18.423us         18.423us         1                1
  rpc_async#aten::mul(worker0 -> worker1)#remote_op: empty          0.14%            15.712us         0.14%            15.712us         15.712us         1                1
  ----------------------------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
  Self CPU time total: 11.237ms

여기서 프로파일러가 ``worker0``\의 ``worker1``\에 대한 ``rpc_async`` 호출을 프로파일링 한것을 확인할 수 있습니다. 특히, 테이블의 처음 2개 항목은 각 RPC호출에 대해 연산자 이름, 원래 작업자(originating worker), 대상 작업자와 같은 세부 정보들을 보여주고 ``CPU total`` 열은 RPC호출의 종단간(end-to-end) 대기 시간을 나타냅니다.

또한 RPC로 worker1에서 원격으로 호출된 실제 연산자를 볼 수 있습니다. ``Node ID`` 열을 확인하여 ``worker1``\에서 발생한 연산들을 확인할 수 있습니다. 예를 들어 ``worker1``\가 입력 텐서에 내장 연산자 ``mul``\을 실행하도록 지정하면서, ``worker0``\에 의해 ``worker1``\에게 전송된 RPC의 결과로 이름이 ``rpc_async#aten::mul(worker0 -> worker1)#remote_op: mul``\인 행을 원격 노드에서 일어나는 ``mul`` 연산으로 해석할 수 있습니다. 원격 연산의 이름은 이를 발생시킨 RPC 이벤트의 이름으로 시작합니다. 예를 들어 ``rpc.rpc_async(dst_worker_name, torch.add, args=(t1, t2))`` 호출에 해당하는 원격 연산은 ``rpc_async#aten::mul(worker0 -> worker1)``\으로 시작합니다.

프로파일러를 사용하면 RPC를 통해 실행된 사용자 정의 함수의 통찰력(insight)을 얻을 수도 있습니다. 예를 들어 위 ``worker`` 함수에 다음을 추가합니다.

::

  # worker() 함수 외부에서 정의
  def udf_with_ops():
      import time
      time.sleep(1)
      t1, t2 = random_tensor(), random_tensor()
      torch.add(t1, t2)
      torch.mul(t1, t2)

  def worker(rank, world_size):
      # 위 코드 생략
      with profiler.profile() as p:
          fut = rpc.rpc_async(dst_worker_name, udf_with_ops, args=())
          fut.wait()
      print(p.key_averages().table())

위 코드는 1초동안 sleep 하는 사용자 정의 함수를 생성하고 다양한 연산자를 실행합니다. 우리가 위에서 수행했던 것과 유사하게, 사용자 정의 함수를 실행하도록 지정하여 원격 worker에 RPC를 보냅니다. 이 코드를 실행하면 다음과 같은 출력을 얻을수 있습니다. 

::

  # 정확한 출력은 무작위성을 필요로 합니다 
  --------------------------------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
  Name                                                                  Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls  Node ID
  --------------------------------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
  rpc_async#udf_with_ops(worker0 -> worker1)                            0.00%            0.000us          0                1.008s           1.008s           1                0
  rpc_async#udf_with_ops(worker0 -> worker1)#remote_op: rand            12.58%           80.037us         47.09%           299.589us        149.795us        2                1
  rpc_async#udf_with_ops(worker0 -> worker1)#remote_op: empty           15.40%           98.013us         15.40%           98.013us         24.503us         4                1
  rpc_async#udf_with_ops(worker0 -> worker1)#remote_op: uniform_        22.85%           145.358us        23.87%           151.870us        75.935us         2                1
  rpc_async#udf_with_ops(worker0 -> worker1)#remote_op: is_complex      1.02%            6.512us          1.02%            6.512us          3.256us          2                1
  rpc_async#udf_with_ops(worker0 -> worker1)#remote_op: add             25.80%           164.179us        28.43%           180.867us        180.867us        1                1
  rpc_async#udf_with_ops(worker0 -> worker1)#remote_op: mul             20.48%           130.293us        31.43%           199.949us        99.975us         2                1
  rpc_async#udf_with_ops(worker0 -> worker1)#remote_op: output_nr       0.71%            4.506us          0.71%            4.506us          2.253us          2                1
  rpc_async#udf_with_ops(worker0 -> worker1)#remote_op: is_leaf         1.16%            7.367us          1.16%            7.367us          1.842us          4                1
  --------------------------------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------

여기서 우리는 사용자 정의 함수가 ``(rpc_async#udf_with_ops(worker0 -> worker1))``\의 이름으로 성공적으로 프로파일링 되고 대략적으로 예상 가능한 총 CPU 시간(주어진 ``sleep`` 1초보다 약간 큰)을 가지는 것을 확인할 수 있습니다. 위의 프로파일링 출력과 유사하게, 이 RPC 요청을 실행하는 과정으로 worker1에서 실행된 원격 연산자을 볼 수 있습니다.

마지막으로, 프로파일러에 의해 제공된 추적(tracing) 기능을 사용하여 원격 실행을 시각화할 수 있습니다. 위 ``worker`` 함수에 다음 코드를 추가합니다.

::

    def worker(rank, world_size):
        # 위 코드 생략
        # 위 프로파일링 출력에 대한 추적 생성
        trace_file = "/tmp/trace.json"
        prof.export_chrome_trace(trace_file)
        logger.debug(f"Wrote trace to {trace_file}")

이제 Chrome에서 추적파일(``chrome://tracing``)을 가져올 수 있습니다. 다음과 유사한 출력이 보여야 합니다.

.. image:: ../_static/img/rpc_trace_img.png
   :scale: 25 %

보시다시피 우리는 RPC 요청을 추적하였으며, 원격 연산의 추적을 시각화 할 수도 있습니다. (이 경우 ``node_id: 1``\에 대한 추적 열에서 주어진)

이것들을 모두 합쳐 이 레시피에 대해 다음과 같은 코드를 얻습니다.

::

    import torch
    import torch.distributed.rpc as rpc
    import torch.autograd.profiler as profiler
    import torch.multiprocessing as mp
    import os
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger()

    def random_tensor():
      return torch.rand((3, 3), requires_grad=True)

    def udf_with_ops():
      import time
      time.sleep(1)
      t1, t2 = random_tensor(), random_tensor()
      torch.add(t1, t2)
      torch.mul(t1, t2)

    def worker(rank, world_size):
      os.environ["MASTER_ADDR"] = "localhost"
      os.environ["MASTER_PORT"] = "29500"
      worker_name = f"worker{rank}"

      # RPC framework 초기화
      rpc.init_rpc(
          name=worker_name,
          rank=rank,
          world_size=world_size
      )
      logger.debug(f"{worker_name} successfully initialized RPC.")

      if rank == 0:
        dst_worker_rank = (rank + 1) % world_size
        dst_worker_name = f"worker{dst_worker_rank}"
        t1, t2 = random_tensor(), random_tensor()
        # 프로파일링 범위하에서 전송하고 RPC 완료를 기다립니다
        with profiler.profile() as prof:
            fut1 = rpc.rpc_async(dst_worker_name, torch.add, args=(t1, t2))
            fut2 = rpc.rpc_async(dst_worker_name, torch.mul, args=(t1, t2))
            # RPC는 프로파일링 범위안에서 대기해야 합니다
            fut1.wait()
            fut2.wait()
        print(prof.key_averages().table())

        with profiler.profile() as p:
            fut = rpc.rpc_async(dst_worker_name, udf_with_ops, args=())
            fut.wait()

        print(p.key_averages().table())

        trace_file = "/tmp/trace.json"
        prof.export_chrome_trace(trace_file)
        logger.debug(f"Wrote trace to {trace_file}")


      logger.debug(f"Rank {rank} waiting for workers and shutting down RPC")
      rpc.shutdown()
      logger.debug(f"Rank {rank} shutdown RPC")



    if __name__ == '__main__':
      # 2개 RPC workers 실행
      world_size = 2
      mp.spawn(worker, args=(world_size,), nprocs=world_size)


더 많은 학습
-------------------

-  설치 지침과 추가적인 문서 `pytorch.org`_
-  RPC framework와 API 참조문서 `Distributed RPC Framework`_
-  프로파일러 문서 `Full profiler documentation`_

.. _pytorch.org: https://pytorch.org/
.. _Full profiler documentation: https://pytorch.org/docs/stable/autograd.html#profiler
.. _Pytorch Profiler: https://pytorch.org/docs/stable/autograd.html#profiler
.. _Distributed RPC Framework: https://pytorch.org/docs/stable/rpc.html
.. _RPC Tutorials: https://pytorch.org/tutorials/intermediate/rpc_tutorial.html
.. _Profiler Recipe: https://pytorch.org/tutorials/recipes/recipes/profiler.html
