Cpp 확장을 사용하여 프로세스 그룹 백엔드 사용자 정의
=====================================================

**저자**: `Feng Tian <https://github.com/ftian1>`__, `Shen Li <https://mrshenli.github.io/>`__, `Min Si <https://minsii.github.io/>`__

.. note::
   |edit| 이 튜토리얼의 소스 코드는 `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/process_group_cpp_extension_tutorial.rst>`__에서 확인하고 변경해 볼 수 있습니다..

선수과목(Prerequisites):

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `PyTorch Collective Communication Package <https://pytorch.org/docs/stable/distributed.html>`__
-  `PyTorch Cpp Extension <https://pytorch.org/docs/stable/cpp_extension.html>`__
-  `Writing Distributed Applications with PyTorch <https://tutorials.pytorch.kr/intermediate/dist_tuto.html>`__

이 튜토리얼은 `cpp 확장 <https://pytorch.org/docs/stable/cpp_extension.html>`__을 사용하여 사용자 정의 ProcessGroup 백엔드를 구현하고 이를 `파이토치 분산 패키지 <https://pytorch.org/docs/stable/distributed.html>`__에 연결하는 방법을 보여줍니다.
이것은 하드웨어에 특화된 소프트웨어 스택이 필요한 경우나 새로운 집합 통신 알고리즘을 실험하고자 할 때 유용합니다.


기초
------

파이토치 집합 통신은 
`분산 데이터 병렬(DistributedDataParallel) <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__,
`제로 리던던시 최적화기(ZeroRedundancyOptimizer) <https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer>`__,
`완전 공유 데이터 병렬(FullyShardedDataParallel) <https://github.com/pytorch/pytorch/blob/master/torch/distributed/_fsdp/fully_sharded_data_parallel.py>`__을 포함한 널리 사용되는 분산 훈련 기능을 지원합니다.
동일한 집합 통신 API를 다양한 통신 백엔드에서 작동하도록 하기 위해 분산 패키지는 집합 통신 작업을 
`ProcessGroup <https://github.com/pytorch/pytorch/blob/release/1.10/torch/csrc/distributed/c10d/ProcessGroup.hpp>`__
클래스로 추상화합니다. 이후에는 원하는 서드파티 라이브러리를 사용하여 ``ProcessGroup``의 하위 클래스로 다양한 백엔드를 구현할 수 있습니다.
파이토치 분산에는 세 가지 기본 백엔드인 ``ProcessGroupNCCL``, ``ProcessGroupGloo``, 그리고 ``ProcessGroupMPI``가 포함되어 있습니다.
그러나 그러나 이 세 가지 백엔드 외에도 다른 통신 라이브러리(예: `UCC <https://github.com/openucx/ucc>`__, `OneCCL <https://github.com/oneapi-src/oneCCL>`__), 다른 유형의 하드웨어(예: `TPU <https://cloud.google.com/tpu>`__, `Trainum <https://aws.amazon.com/machine-learning/trainium/>`__), 
그리고 새로운 통신 알고리즘(예: `Herring <https://www.amazon.science/publications/herring-rethinking-the-parameter-server-at-scale-for-the-cloud>`__, `Reduction Server <https://cloud.google.com/blog/topics/developers-practitioners/optimize-training-performance-reduction-server-vertex-ai>`__)도 있습니다.
따라서 분산 패키지는 집합 통신 백엔드를 사용자 지정할 수 있도록 확장 API를 노출합니다.


아래의 4단계는 더미 ``ProcessGroup`` 백엔드를 구현하고 파이썬 응용 프로그램 코드에서 사용하는 방법을 보여줍니다.
이 튜토리얼은 작동하는 통신 백엔드를 개발하는 대신 확장 API를 설명하는 데 중점을 둡니다. 따라서 ``dummy`` 백엔드는 API의 일부 (``all_reduce`` 및 ``all_gather``)를 다루며 텐서의 값을 단순히 0으로 설정합니다.


Step 1: Implement a Subclass of ``ProcessGroup``
------------------------------------------------

This first step is to implement a ``ProcessGroup`` subclass that overrides
target collective communication APIs and runs the custom communication algorithm.
The extension also needs to implement a ``Work`` subclass, which
serves as a future of communication results and allows asynchronous execution in
application code. If the extension uses third-party libraries, it can
include the headers and call into the library APIs from the ``ProcessGroupDummy``
subclass. The two code snippets below present the implementation of ``dummy.h`` and
``dummy.cpp``. See the `dummy collectives <https://github.com/mrshenli/dummy_collectives>`__
repository for the full implementation.

.. code-block:: cpp

    // file name: dummy.hpp
    #include <torch/python.h>

    #include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
    #include <torch/csrc/distributed/c10d/Work.hpp>
    #include <torch/csrc/distributed/c10d/Store.hpp>
    #include <torch/csrc/distributed/c10d/Types.hpp>
    #include <torch/csrc/distributed/c10d/Utils.hpp>

    #include <pybind11/chrono.h>

    namespace c10d {

    class ProcessGroupDummy : public ProcessGroup {
      public:
        ProcessGroupDummy(int rank, int size);

        c10::intrusive_ptr<Work> allgather(
            std::vector<std::vector<at::Tensor>>& outputTensors,
            std::vector<at::Tensor>& inputTensors,
            const AllgatherOptions& opts = AllgatherOptions()) override;

        c10::intrusive_ptr<Work> allreduce(
            std::vector<at::Tensor>& tensors,
            const AllreduceOptions& opts = AllreduceOptions()) override;

        // The collective communication APIs without a custom implementation
        // will error out if invoked by application code.
    };

    class WorkDummy : public Work {
      public:
        WorkDummy(
          OpType opType,
          c10::intrusive_ptr<c10::ivalue::Future> future) // future of the output
          : Work(
              -1, // rank, only used by recvAnySource, irrelevant in this demo
              opType),
          future_(std::move(future)) {}
        // There are several additional helper functions that need to be
        // implemented. Please refer to https://github.com/mrshenli/dummy_collectives
        // for the full implementation.

      private:
        c10::intrusive_ptr<c10::ivalue::Future> future_;
    };
    } // namespace c10d


.. code-block:: cpp

    // file name: dummy.cpp
    #include "dummy.hpp"

    namespace c10d {

    // This is a dummy allgather that sets all output tensors to zero
    // Modify the implementation to conduct real communication asynchronously
    c10::intrusive_ptr<Work> ProcessGroupDummy::allgather(
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

    // This is a dummy allreduce that sets all output tensors to zero
    // Modify the implementation to conduct real communication asynchronously
    c10::intrusive_ptr<Work> ProcessGroupDummy::allreduce(
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

Step 2: Expose The Extension Python APIs
----------------------------------------

The backend constructors are called
`from Python side <https://github.com/pytorch/pytorch/blob/v1.9.0/torch/distributed/distributed_c10d.py#L643-L650>`__,
so the extension also needs to expose the constructor APIs to Python. This can
be done by adding the following methods. In this example, ``store`` and
``timeout`` are ignored by the ``ProcessGroupDummy`` instantiation method, as
those are not used in this dummy implementation. However, real-world extensions
should consider using the ``store`` to perform rendezvous and supporting the
``timeout`` argument.

.. code-block:: cpp

    class ProcessGroupDummy : public ProcessGroup {
        static c10::intrusive_ptr<ProcessGroup> createProcessGroupDummy(
            const c10::intrusive_ptr<::c10d::Store>& store,
            int rank,
            int size,
            const std::chrono::duration<float>& timeout);

        static void ProcessGroupDummyConstructor() __attribute__((constructor)) {
            py::object module = py::module::import("torch.distributed");
            py::object register_backend =
                module.attr("Backend").attr("register_backend");
            // torch.distributed.Backend.register_backend will add `dummy` as a
            // new valid backend.
            register_backend("dummy", py::cpp_function(createProcessGroupDummy));
        }
    }

.. code-block:: cpp

    c10::intrusive_ptr<ProcessGroup> ProcessGroupDummy::createProcessGroupDummy(
            const c10::intrusive_ptr<::c10d::Store>& /* unused */,
            int rank,
            int size,
            const std::chrono::duration<float>& /* unused */) {
        return c10::make_intrusive<ProcessGroupDummy>(rank, size);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("createProcessGroupDummy", &ProcessGroupDummy::createProcessGroupDummy);
    }


Step 3: Build The Custom Extension
----------------------------------

Now, the extension source code files are ready. We can then use
`cpp extensions <https://pytorch.org/docs/stable/cpp_extension.html>`__
to build it. To do that, create a ``setup.py`` file that prepares the paths and
commands. Then call ``python setup.py install`` to install the extension.

If the extension depends on third-party libraries, you can also specify
``libraries_dirs`` and ``libraries`` to the cpp extension APIs. See the
`torch ucc <https://github.com/openucx/torch-ucc>`__
project as a real-world example.

.. code-block:: python

    # file name: setup.py
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

Step 4: Use The Extension in Application
----------------------------------------

After installation, you can conveniently use the ``dummy`` backend when calling
`init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__
as if it is an builtin backend.

.. code-block:: python

    import os

    import torch
    # importing dummy_collectives makes torch.distributed recognize `dummy`
    # as a valid backend.
    import dummy_collectives

    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("dummy", rank=0, world_size=1)

    x = torch.ones(6)
    dist.all_reduce(x)
    print(f"cpu allreduce: {x}")
    if torch.cuda.is_available():
        y = x.cuda()
        dist.all_reduce(y)
        print(f"cuda allreduce: {y}")

    try:
        dist.broadcast(x, 0)
    except RuntimeError:
        print("got RuntimeError as broadcast is not implemented in Dummy ProcessGroup")
