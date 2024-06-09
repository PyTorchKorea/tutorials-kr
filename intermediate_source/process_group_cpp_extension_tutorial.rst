Cpp 확장을 사용한 프로세스 그룹 백엔드 사용자 정의
=======================================================

**Author**: `Howard Huang <https://github.com/H-Huang>`__, `Feng Tian <https://github.com/ftian1>`__, `Shen Li <https://mrshenli.github.io/>`__, `Min Si <https://minsii.github.io/>`__
  **번역**: `박재윤 <https://github.com/jenner9212>`_

.. note::
   |edit| 이 튜토리얼의 소스 코드는 `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/process_group_cpp_extension_tutorial.rst>`__ 에서 확인하고 변경해 볼 수 있습니다.

선수과목(Prerequisites):

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `PyTorch Collective Communication Package <https://pytorch.org/docs/stable/distributed.html>`__
-  `PyTorch Cpp Extension <https://pytorch.org/docs/stable/cpp_extension.html>`__
-  `Writing Distributed Applications with PyTorch <https://tutorials.pytorch.kr/intermediate/dist_tuto.html>`__

이 튜토리얼에서는 `cpp 확장 <https://pytorch.org/docs/stable/cpp_extension.html>`__ 을 사용하여
사용자 정의 ``Backend`` 를 구현하고 이를 `파이토치 분산 패키지 <https://pytorch.org/docs/stable/distributed.html>`__ 에
어떻게 연결하는지를 알아봅니다.
이러한 방법은 하드웨어에 특화된 소프트웨어 스택이 필요한 경우나 새로운 집합 통신 알고리즘(collective communication algorithm)을
실험하고자 할 때 유용합니다.


기초
------

파이토치(PyTorch)의 집합 통신(collective communications)은
`분산 데이터 병렬(DistributedDataParallel) <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__,
`제로 리던던시 최적화기(ZeroRedundancyOptimizer) <https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer>`__,
`완전 공유 데이터 병렬(FullyShardedDataParallel) <https://github.com/pytorch/pytorch/blob/master/torch/distributed/_fsdp/fully_sharded_data_parallel.py>`__
등을 포함하여, 널리 사용되는 분산 학습 기능을 지원합니다.
동일한 집합 통신 API를 다양한 통신 백엔드에서 작동하도록 하기 위해 분산 패키지는 집합 통신 작업을
`Backend <https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp>`__
클래스로 추상화합니다. 이후에는 원하는 서드파티 라이브러리를 사용하여
``Backend`` 의 하위 클래스(subclass)로 다양한 백엔드를 구현할 수 있습니다.
파이토치 분산(PyTorch distributed)에는 세 가지 기본 백엔드인
``ProcessGroupNCCL``, ``ProcessGroupGloo``, 그리고 ``ProcessGroupMPI`` 가 포함되어 있습니다.
그러나 이 세 가지 백엔드 외에도 다른 통신 라이브러리(예: `UCC <https://github.com/openucx/ucc>`__, `OneCCL <https://github.com/oneapi-src/oneCCL>`__), 다른 유형의 하드웨어(예: `TPU <https://cloud.google.com/tpu>`__, `Trainum <https://aws.amazon.com/machine-learning/trainium/>`__),
그리고 새로운 통신 알고리즘(예: `Herring <https://www.amazon.science/publications/herring-rethinking-the-parameter-server-at-scale-for-the-cloud>`__, `Reduction Server <https://cloud.google.com/blog/topics/developers-practitioners/optimize-training-performance-reduction-server-vertex-ai>`__)도 있습니다.
따라서 분산 패키지는 집합 통신 백엔드를 사용자 지정할 수 있도록 확장 API를 노출합니다.


아래의 4단계는 가짜(dunmmy) ``ProcessGroup`` 백엔드를 구현하고 파이썬 응용 프로그램 코드에서 사용하는 방법을 보여줍니다.
이 튜토리얼은 작동하는 통신 백엔드를 개발하는 대신 확장 API를 설명하는 데 중점을 둡니다. 따라서 ``dummy`` 백엔드는 API의 일부 (``all_reduce`` 및 ``all_gather``)를 다루며 tensor의 값을 단순히 0으로 설정합니다.


단계 1: ``Backend`` 의 하위 클래스 구현
------------------------------------------------

첫 번째 단계는 대상 집합 통신 API를 재정의하고 사용자 정의 통신 알고리즘을 실행하는 ``Backend`` 하위 클래스를 구현하는 것입니다.
확장 기능은 미래(future) 통신 결과를 제공하는 ``Work`` 하위 클래스를 구현해야 하며, 이는 응용 프로그램 코드에서 비동기 실행을 허용합니다.
확장 기능이 서드파티 라이브러리를 사용하는 경우, 해당 확장 기능은 ``BackendDemmy`` 하위 클래스에서 헤더를 포함하고 라이브러리 API를 호출할 수 있습니다.
아래의 두 코드는 ``dummy.h`` 및 ``dummy.cpp`` 의 구현을 보여줍니다. 전체 구현은 `더미 집합(dummy collectives) <https://github.com/H-Huang/torch_collective_extension>`__ 저장소에서 확인하실 수 있습니다.

.. code-block:: cpp

    // 파일 이름: dummy.hpp
    #include <torch/python.h>

    #include <torch/csrc/distributed/c10d/Backend.hpp>
    #include <torch/csrc/distributed/c10d/Work.hpp>
    #include <torch/csrc/distributed/c10d/Store.hpp>
    #include <torch/csrc/distributed/c10d/Types.hpp>
    #include <torch/csrc/distributed/c10d/Utils.hpp>

    #include <pybind11/chrono.h>

    namespace c10d {

    class BackendDummy : public Backend {
      public:
        BackendDummy(int rank, int size);

        c10::intrusive_ptr<Work> allgather(
            std::vector<std::vector<at::Tensor>>& outputTensors,
            std::vector<at::Tensor>& inputTensors,
            const AllgatherOptions& opts = AllgatherOptions()) override;

        c10::intrusive_ptr<Work> allreduce(
            std::vector<at::Tensor>& tensors,
            const AllreduceOptions& opts = AllreduceOptions()) override;

        // 사용자 정의 구현이 없는 상태에서의 집합 통신 API는
        // 응용 프로그램 코드에서 호출되면 오류가 발생합니다.
    };

    class WorkDummy : public Work {
      public:
        WorkDummy(
          OpType opType,
          c10::intrusive_ptr<c10::ivalue::Future> future) // future of the output
          : Work(
              -1, // 랭크, recvAnySource에서만 사용되며 이 데모에서는 관련이 없습니다.
              opType),
          future_(std::move(future)) {}
        bool isCompleted() override;
        bool isSuccess() const override;
        bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
        virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

      private:
        c10::intrusive_ptr<c10::ivalue::Future> future_;
    };
    } // namespace c10d


.. code-block:: cpp

    // 파일 이름: dummy.cpp
    #include "dummy.hpp"

    namespace c10d {

    // 이것은 모든 출력 tensor를 0으로 설정하는 가짜(dummy) allgather입니다.
    // 실제 통신을 비동기적으로 수행하도록 구현을 수정하세요.
    c10::intrusive_ptr<Work> BackendDummy::allgather(
            std::vector<std::vector<at::Tensor>>& outputTensors,
            std::vector<at::Tensor>& inputTensors,
            const AllgatherOptions& /* unused */) {
        for (auto& outputTensorVec : outputTensors) {
            for (auto& outputTensor : outputTensorVec) {
                outputTensor.zero_();
            }
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
        future->markCompleted(c10::IValue(outputTensors));
        return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
    }

    // 이것은 모든 출력 tensor를 0으로 설정하는 가짜(dummy) allreduce입니다.
    // 실제 통신을 비동기적으로 수행하도록 구현을 수정하세요.
    c10::intrusive_ptr<Work> BackendDummy::allreduce(
            std::vector<at::Tensor>& tensors,
            const AllreduceOptions& opts) {
        for (auto& tensor : tensors) {
            tensor.zero_();
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
    }
    } // namespace c10d

단계 2: 확장 기능을 파이썬 API로 노출
------------------------------------------

백엔드 생성자는 `파이썬 측 <https://github.com/pytorch/pytorch/blob/v1.9.0/torch/distributed/distributed_c10d.py#L643-L650>`__ 에서
호출되므로 확장 기능도 파이썬에 생성자 API를 노출해야 합니다.
다음 메서드를 추가함으로써 이 작업을 수행할 수 있습니다.
이 예제에서는 ``store`` 와 ``timeout`` 이 사용되지 않으므로 ``BackendDummy`` 인스턴스화 메서드에서 무시됩니다.
그러나 실제 확장 기능은 랑데뷰를 수행하고 ``timeout`` 인수를 지원하기 위해 ``store`` 사용을 고려해야 합니다.

.. code-block:: cpp

    // file name: dummy.hpp
    class BackendDummy : public Backend {
        ...
        <Step 1 code>
        ...

        static c10::intrusive_ptr<Backend> createBackendDummy(
            const c10::intrusive_ptr<::c10d::Store>& store,
            int rank,
            int size,
            const std::chrono::duration<float>& timeout);

        static void BackendDummyConstructor() __attribute__((constructor)) {
            py::object module = py::module::import("torch.distributed");
            py::object register_backend =
                module.attr("Backend").attr("register_backend");
            // torch.distributed.Backend.register_backend는
            // `dummy` 를 새로운 유효한 백엔드로 추가합니다.
            register_backend("dummy", py::cpp_function(createProcessGroupDummy));
        }
    }

.. code-block:: cpp

    // file name: dummy.cpp
    c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
            const c10::intrusive_ptr<::c10d::Store>& /* unused */,
            int rank,
            int size,
            const std::chrono::duration<float>& /* unused */) {
        return c10::make_intrusive<BackendDummy>(rank, size);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("createBackendDummy", &BackendDummy::createBackendDummy);
    }


단계 3: 사용자 정의 확장 빌드
------------------------------------

이제 확장 소스 코드 파일이 준비되었습니다. 그런 다음 `cpp 확장 <https://pytorch.org/docs/stable/cpp_extension.html>`__ 을 사용하여 빌드할 수 있습니다.
이를 위해 경로와 명령을 준비하는 ``setup.py`` 파일을 생성하고, ``python setup.py develop`` 을 호출하여 확장을 설치합니다.

확장이 서드파티 라이브러리에 의존하는 경우, cpp 확장 API에 ``libraries_dirs`` 및 ``libraries`` 지정할 수도 있습니다. 실제 예제로 `torch ucc <https://github.com/openucx/torch-ucc>`__ 프로젝트를 참조하십시오.

.. code-block:: python

    # 파일 이름: setup.py
    import os
    import sys
    import torch
    from setuptools import setup
    from torch.utils import cpp_extension

    sources = ["src/dummy.cpp"]
    include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/"]

    if torch.cuda.is_available():
        module = cpp_extension.CUDAExtension(
            name = "dummy_collectives",
            sources = sources,
            include_dirs = include_dirs,
        )
    else:
        module = cpp_extension.CppExtension(
            name = "dummy_collectives",
            sources = sources,
            include_dirs = include_dirs,
        )

    setup(
        name = "Dummy-Collectives",
        version = "0.0.1",
        ext_modules = [module],
        cmdclass={'build_ext': cpp_extension.BuildExtension}
    )

단계 4: 응용 프로그램에서 확장 기능 사용
--------------------------------------------

설치 후 `init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__ 을 호출할 때 ``dummy`` 백엔드를 내장된 백엔드처럼 편리하게 사용할 수 있습니다.

``init_process_group`` 의 ``backend`` 인자(argument)를 ``dummy`` 로 변경하여 백엔드를 기반으로 디스패치(dispatch)하도록 지정할 수 있습니다.
이 때 ``backend`` 인자로 ``cpu:gloo,cuda:dummy`` 를 지정하면 CPU 텐서에 대해서는 ``gloo`` 백엔드를 사용하고 CUDA 텐서에 대해서는 ``dummy`` 백엔드를 사용하여
집합 통신을 디스패치하도록 지정합니다.

모든 텐서들을 ``dummy`` 백엔드로 보내려면 그냥 ``dummy`` 를 백엔드 인자로 지정하면 됩니다.

.. code-block:: python

    import os

    import torch
    # dummy_collectives를 import하면 torch.distributed가 `dummy` 를 유효한 백엔드로 인식합니다.
    import dummy_collectives

    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Alternatively:
    # dist.init_process_group("dummy", rank=0, world_size=1)
    dist.init_process_group("cpu:gloo,cuda:dummy", rank=0, world_size=1)

    # 이 텐서는 gloo 백엔드를 사용하고
    x = torch.ones(6)
    dist.all_reduce(x)
    print(f"cpu allreduce: {x}")

    # 이 텐서는 dummy 백엔드를 사용합니다.
    if torch.cuda.is_available():
        y = x.cuda()
        dist.all_reduce(y)
        print(f"cuda allreduce: {y}")

        try:
            dist.broadcast(y, 0)
        except RuntimeError:
            print("got RuntimeError when calling broadcast")
