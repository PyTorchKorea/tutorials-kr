PyTorch C++ API에서 CUDA 그래프 사용하기
===========================================

**번역**: `장효영 <https://github.com/hyoyoung>`_

.. note::
   |edit| 이 튜토리얼을 여기서 보고 편집하세요 `GitHub <https://github.com/pytorch/tutorials/blob/main/advanced_source/cpp_cuda_graphs.rst>`__. 전체 소스 코드는 여기에 있습니다 `GitHub <https://github.com/pytorch/tutorials/blob/main/advanced_source/cpp_cuda_graphs>`__.

선수 지식:

-  `PyTorch C++ 프론트엔드 사용하기 <../advanced_source/cpp_frontend.html>`__
-  `CUDA semantics <https://pytorch.org/docs/master/notes/cuda.html>`__
-  Pytorch 2.0 이상
-  CUDA 11 이상

NVIDIA의 CUDA 그래프는 버전 10 릴리즈 이후로 CUDA 툴킷 라이브러리의 일부였습니다
 `version 10 <https://developer.nvidia.com/blog/cuda-graphs/>`_.
CPU 과부하를 크게 줄여 애플리케이션의 성능을 향상시킵니다.

이 튜토리얼에서는, CUDA 그래프 사용에 초점을 맞출 것입니다
`PyTorch C++ 프론트엔드 사용하기 <https://tutorials.pytorch.kr/advanced/cpp_frontend.html>`_.
C++ 프론트엔드는 파이토치 사용 사례의 중요한 부분인데, 주로 제품 및 배포 애플리케이션에서 활용됩니다.
`첫번째 등장 <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>`_
이후로 CUDA 그래프는 매우 성능이 좋고 사용하기 쉬워서, 사용자와 개발자의 마음을 사로잡았습니다.
실제로, CUDA 그래프는 파이토치 2.0의 ``torch.compile`` 에서 기본적으로 사용되며,
훈련과 추론 시에 생산성을 높여줍니다.

파이토치에서 CUDA 그래프 사용법을 보여드리고자 합니다 `MNIST
예제 <https://github.com/pytorch/examples/tree/main/cpp/mnist>`_.
LibTorch(C++ 프론트엔드)에서의 CUDA 그래프 사용법은 다음과 매우 유사하지만
`Python 사용예제 <https://pytorch.org/docs/main/notes/cuda.html#cuda-graphs>`_
약간의 구문과 기능의 차이가 있습니다.

시작하기
---------------

주요 훈련 루프는 여러 단계로 구성되어 있으며
다음 코드 모음에 설명되어 있습니다.

.. code-block:: cpp

  for (auto& batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    loss.backward();
    optimizer.step();
  }

위의 예시에는 순전파, 역전파, 가중치 업데이트가 포함되어 있습니다.

이 튜토리얼에서는 전체 네트워크 그래프 캡처를 통해 모든 계산 단계에 CUDA 그래프를 적용합니다.
하지만 그 전에 약간의 소스 코드 수정이 필요합니다. 우리가 해야 할 일은 주 훈련 루프에서
tensor를 재사용할 수 있도록 tensor를 미리 할당하는 것입니다.
다음은 구현 예시입니다.

.. code-block:: cpp

  torch::TensorOptions FloatCUDA =
      torch::TensorOptions(device).dtype(torch::kFloat);
  torch::TensorOptions LongCUDA =
      torch::TensorOptions(device).dtype(torch::kLong);

  torch::Tensor data = torch::zeros({kTrainBatchSize, 1, 28, 28}, FloatCUDA);
  torch::Tensor targets = torch::zeros({kTrainBatchSize}, LongCUDA);
  torch::Tensor output = torch::zeros({1}, FloatCUDA);
  torch::Tensor loss = torch::zeros({1}, FloatCUDA);

  for (auto& batch : data_loader) {
    data.copy_(batch.data);
    targets.copy_(batch.target);
    training_step(model, optimizer, data, targets, output, loss);
  }

여기서 ``training_step``은 단순히 해당 옵티마이저 호출과 함께 순전파 및 역전파로 구성됩니다

.. code-block:: cpp

  void training_step(
      Net& model,
      torch::optim::Optimizer& optimizer,
      torch::Tensor& data,
      torch::Tensor& targets,
      torch::Tensor& output,
      torch::Tensor& loss) {
    optimizer.zero_grad();
    output = model.forward(data);
    loss = torch::nll_loss(output, targets);
    loss.backward();
    optimizer.step();
  }

파이토치의 CUDA 그래프 API는 스트림 캡처에 의존하고 있으며, 이 경우 다음처럼 사용됩니다

.. code-block:: cpp

  at::cuda::CUDAGraph graph;
  at::cuda::CUDAStream captureStream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(captureStream);

  graph.capture_begin();
  training_step(model, optimizer, data, targets, output, loss);
  graph.capture_end();

실제 그래프 캡처 전에, 사이드 스트림에서 여러 번의 워밍업 반복을 실행하여
CUDA 캐시뿐만 아니라 훈련 중에 사용할
CUDA 라이브러리(CUBLAS와 CUDNN같은)를 준비하는 것이 중요합니다.

.. code-block:: cpp

  at::cuda::CUDAStream warmupStream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(warmupStream);
  for (int iter = 0; iter < num_warmup_iters; iter++) {
    training_step(model, optimizer, data, targets, output, loss);
  }

그래프 캡처에 성공하면 ``training_step(model, optimizer, data, target, output, loss);`` 호출을
``graph.replay()``로 대체하여 학습 단계를 진행할 수 있습니다.

훈련 결과
----------------

코드를 한 번 살펴보면 그래프가 아닌 일반 훈련에서 다음과 같은 결과를 볼 수 있습니다

.. code-block:: shell

  $ time ./mnist
  Train Epoch: 1 [59584/60000] Loss: 0.3921
  Test set: Average loss: 0.2051 | Accuracy: 0.938
  Train Epoch: 2 [59584/60000] Loss: 0.1826
  Test set: Average loss: 0.1273 | Accuracy: 0.960
  Train Epoch: 3 [59584/60000] Loss: 0.1796
  Test set: Average loss: 0.1012 | Accuracy: 0.968
  Train Epoch: 4 [59584/60000] Loss: 0.1603
  Test set: Average loss: 0.0869 | Accuracy: 0.973
  Train Epoch: 5 [59584/60000] Loss: 0.2315
  Test set: Average loss: 0.0736 | Accuracy: 0.978
  Train Epoch: 6 [59584/60000] Loss: 0.0511
  Test set: Average loss: 0.0704 | Accuracy: 0.977
  Train Epoch: 7 [59584/60000] Loss: 0.0802
  Test set: Average loss: 0.0654 | Accuracy: 0.979
  Train Epoch: 8 [59584/60000] Loss: 0.0774
  Test set: Average loss: 0.0604 | Accuracy: 0.980
  Train Epoch: 9 [59584/60000] Loss: 0.0669
  Test set: Average loss: 0.0544 | Accuracy: 0.984
  Train Epoch: 10 [59584/60000] Loss: 0.0219
  Test set: Average loss: 0.0517 | Accuracy: 0.983

  real    0m44.287s
  user    0m44.018s
  sys    0m1.116s

CUDA 그래프를 사용한 훈련은 다음과 같은 출력을 생성합니다

.. code-block:: shell

  $ time ./mnist --use-train-graph
  Train Epoch: 1 [59584/60000] Loss: 0.4092
  Test set: Average loss: 0.2037 | Accuracy: 0.938
  Train Epoch: 2 [59584/60000] Loss: 0.2039
  Test set: Average loss: 0.1274 | Accuracy: 0.961
  Train Epoch: 3 [59584/60000] Loss: 0.1779
  Test set: Average loss: 0.1017 | Accuracy: 0.968
  Train Epoch: 4 [59584/60000] Loss: 0.1559
  Test set: Average loss: 0.0871 | Accuracy: 0.972
  Train Epoch: 5 [59584/60000] Loss: 0.2240
  Test set: Average loss: 0.0735 | Accuracy: 0.977
  Train Epoch: 6 [59584/60000] Loss: 0.0520
  Test set: Average loss: 0.0710 | Accuracy: 0.978
  Train Epoch: 7 [59584/60000] Loss: 0.0935
  Test set: Average loss: 0.0666 | Accuracy: 0.979
  Train Epoch: 8 [59584/60000] Loss: 0.0744
  Test set: Average loss: 0.0603 | Accuracy: 0.981
  Train Epoch: 9 [59584/60000] Loss: 0.0762
  Test set: Average loss: 0.0547 | Accuracy: 0.983
  Train Epoch: 10 [59584/60000] Loss: 0.0207
  Test set: Average loss: 0.0525 | Accuracy: 0.983

  real    0m6.952s
  user    0m7.048s
  sys    0m0.619s

결론
----------
위 예시에서 볼 수 있듯이, 바로 `MNIST 예제
<https://github.com/pytorch/examples/tree/main/cpp/mnist>`_ 에  CUDA 그래프를 적용하는 것만으로도
훈련 성능을 6배 이상 향상시킬 수 있었습니다.
이렇게 큰 성능 향상이 가능했던 것은 모델 크기가 작았기 때문입니다.
GPU 사용량이 많은 대형 모델의 경우 CPU 과부하의 영향이 적기 때문에 개선 효과가 더 작을 수 있습니다.
그런 경우라도, GPU의 성능을 이끌어내려면 CUDA 그래프를 사용하는 것이 항상 유리합니다.
