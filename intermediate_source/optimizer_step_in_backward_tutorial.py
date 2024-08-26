"""

How to save memory by fusing the optimizer step into the backward pass
======================================================================

Hello there! This tutorial aims to showcase one way of reducing the
memory footprint of a training loop by reducing the memory taken by
the *gradients*. Say you have a model and you're interested in ways to
optimize memory to avoid ``Out of Memory`` (OOM) errors or simply to ooze
more out of your GPU. Well, you _might_ be in luck (if gradients take up
a portion of your memory and you do not need to do gradient accumulation).
We will explore the following:

1. What takes up memory during your training or finetuning loop,
2. How to capture and visualize memory snapshots to determine the bottleneck,
3. The new ``Tensor.register_post_accumulate_grad_hook(hook)`` API, and finally,
4. How everything fits together in 10 lines to achieve memory savings.

To run this tutorial, you will need:

*  PyTorch 2.1.0 or newer with ``torchvision``
*  1 CUDA GPU if you'd like to run the memory visualizations locally.
   Otherwise, this technique would benefit similarly on any device.

Let us start by importing the required modules and models. We will use a
vision transformer model from torchvision, but feel free to substitute
with your own model. We will also use ``torch.optim.Adam`` as our optimizer,
but, again, feel free to substitute with your own optimizer.

"""

import torch
from torchvision import models
from pickle import dump

model = models.vit_l_16(weights='DEFAULT').cuda()
optimizer = torch.optim.Adam(model.parameters())

###############################################################################
# 이제 일반적인 학습 루프를 정의해봅시다. 실제 학습 시에는 진짜 이미지를 사용해야 
# 하지만, 이 튜토리얼에서는 가짜 입력 데이터를 사용하며 
# 실제 데이터를 로드하는 것에 대해서는 신경 쓰지 않을 것입니다.

IMAGE_SIZE = 224

def train(model, optimizer):
  # 가짜 이미지 입력값 생성: tensor의 형태는 batch_size, channels, height, width
  fake_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()

  # 순전파(forward)와 역전파(backward) 호출
  loss = model.forward(fake_image)
  loss.sum().backward()

  # 옵티마이저 업데이트
  optimizer.step()
  optimizer.zero_grad()

###############################################################################
# 학습 중의 메모리 사용량
# """"""""""""""""""""""""""""
# 이제 메모리 스냅샷을 확인하려고 하므로, 이를 적절히 분석할 준비를 해야 합니다.
# 일반적으로 학습 메모리는 다음으로 구성됩니다:
#
#  * 모델 파라미터 (크기 P)
#  * 역전파 단계를 위해 저장된 활성화 값들(activations) (크기 A)
#  * 변화도, 모델 파라미터와 같은 크기이므로 크기 G = P.
#  * 옵티마이저 상태, 파라미터 크기에 비례함. 예시의 경우, 
#    Adam의 상태는 모델 파라미터의 2배가 필요하므로 크기 O = 2P.
#  * 중간 단계(Intermediate) tensor, 계산 도중 할당됩니다. 
#    보통 크기가 작고 일시적이므로 지금은 신경 쓰지 않겠습니다.
#
# 메모리 스냅샷 캡처 및 시각화
# """"""""""""""""""""""""""""""""""""""""""
# 이제 메모리 스냅샷을 가져와 봅시다! 코드를 실행되는 동안,
# CUDA 메모리 타임라인이 어떤 모습일지 한 번 예상해 보세요.

# CUDA에 메모리 할당 기록을 시작하도록 지시
torch.cuda.memory._record_memory_history(enabled='all')

# 학습 3회 실시
for _ in range(3):
  train(model, optimizer)

# 메모리 할당 스냅샷을 저장
s = torch.cuda.memory._snapshot()
with open(f"snapshot.pickle", "wb") as f:
    dump(s, f)

# CUDA에 메모리 할당 기록을 중지하도록 지시
torch.cuda.memory._record_memory_history(enabled=None)

###############################################################################
# 이제 CUDA 메모리 시각화 도구(CUDA Memory Visualizer)에서 스냅샷을 열어보세요.
# https://pytorch.org/memory_viz 로 들어가서 ``snapshot.pickle`` 파일을 드래그 앤
# 드롭하여 업로드할 수 있습니다. 메모리 타임라인이 예상과 일치하나요?
# 
# .. figure:: /_static/img/optim_step_in_bwd/snapshot.jpg
#    :alt: snapshot.png loaded into CUDA Memory Visualizer
# 
# The model parameters have already been loaded in memory before the training
# step, so we see a chunk of memory devoted to the weights right off the bat.
# As we start our forward pass, memory is allocated gradually for the activations,
# or the tensors we are saving to be able to compute gradients in the backward pass.
# Once we start the backward pass, the activations are gradually freed while memory
# of the gradients starts building up.
# 
# Lastly, as the optimizer kicks in, its state will be lazily initialized, so we 
# should see the optimizer state memory gradually increase during the optimizer
# step of the first training loop only. In future loops, the optimizer memory
# will remain and be updated in-place. The memory for the gradients is then
# freed accordingly at the end of every training loop when ``zero_grad`` is called.
# 
# 이 학습 루프에서 메모리 병목 현상이 발생하는 지점은 어디일까요? 즉, 메모리 
# 사용이 가장 높은 지점은 어디일까요?
# 
# The peak memory usage is during the optimizer step! Note the memory then
# consists of ~1.2GB of parameters, ~1.2GB of gradients, and ~2.4GB=2*1.2GB of
# the optimizer state as expected. The last ~1.2GB comes from Adam optimizer
# requiring memory for intermediates, totaling to ~6GB of peak memory.
# Technically, you can remove the need for the last 1.2GB for optimizer
# intermediates if you set ``Adam(model.parameters(), foreach=False)`` which
# would trade off runtime for memory. If switching off the ``foreach`` runtime
# optimization is sufficient in memory savings for you, nice, but please
# read on if you're curious how this tutorial can help you do better!
# With the technique we will soon introduce, we will reduce peak memory by
# removing the need for the ~1.2GB of **gradients memory** as well as **optimizer
# intermediates memory**. Now, what would you expect the new peak memory to be?
# The answer will be revealed in the `next` snapshot.
#
# 주의 사항: 이 방법은 모든 경우에 적합한 것은 **아님**
# """""""""""""""""""""""""""""""""""""""""""""
# 너무 흥분하기 전에, 먼저 이 방법이 `당신` 의 사용 사례에 적합한지 고려해야 합니다.
# 이 방법은 결코 만능 해결책이 아닙니다! 
# 옵티마이저 단계를 역전파 과정에 합치는 이 방법은 *변화도* 메모리의 감소만을 목표로 
# 합니다 (그리고 부수적으로 옵티마이저 중간 단계 메모리도 줄입니다). 
# 따라서 변화도가 차지하는 메모리가 클수록, 메모리 절감 효과가 더욱 커집니다. 
# 위의 예시에서 변화도는 메모리 총량의 20%를 차지하는데, 이는 꽤나 큰 비율이죠!
#
# 그러나 때에 따라 이러한 상황에 해당하지 않을 수 있습니다. 예를 들어, 이미 
# 가중치가 매우 작다면 (LoRa 적용 등의 이유로), 변화도가 학습 루프에서 공간을 많이 
# 차지하지 않을 것이고, 그렇다면 이 방법의 이점이 그다지 크지 않을 수 있습니다.
# 이런 경우에는 먼저 활성화 체크포인팅, 분산 학습, 양자화, 배치 크기 축소와 같은 
# 다른 기술을 시도해 보세요. 그런 다음, 변화도가 다시 병목의 일부가 될 때 
# 이 튜토리얼로 돌아오세요!
# 
# 아직 여기에 계신가요? 좋습니다, 이제 Tensor의 새로운 ``register_post_accumulate_grad_hook(hook)``
# API를 소개하겠습니다.
#
# ``Tensor.register_post_accumulate_grad_hook(hook)`` API and our technique
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Our technique relies on not having to save the gradients during ``backward()``. Instead,
# once a gradient has been accumulated, we will immediately apply the optimizer to
# the corresponding parameter and drop that gradient entirely! This removes the need
# for holding onto a big buffer of gradients until the optimizer step.
#
# So how can we unlock the behavior of applying the optimizer more eagerly? In our 2.1
# release, we've added a new API :func:`torch.Tensor.register_post_accumulate_grad_hook`
# that would allow us to add a hook onto a Tensor once its ``.grad`` field has been
# accumulated. We will encapsulate the optimizer step into this hook. How?
# 
# How everything fits together in 10 lines
# """"""""""""""""""""""""""""""""""""""""
# Remember our model and optimizer setup from the beginning? I'll leave them commented
# out below so we don't spend resources rerunning the code.
#
# .. code-block:: python
#
#    model = models.vit_l_16(weights='DEFAULT').cuda()
#    optimizer = torch.optim.Adam(model.parameters())

# Instead of having just *one* optimizer, we will have a ``dict`` of optimizers
# for every parameter so we could reference them in our hook.
optimizer_dict = {p: torch.optim.Adam([p], foreach=False) for p in model.parameters()}

# Define our hook, which will call the optimizer ``step()`` and ``zero_grad()``
def optimizer_hook(parameter) -> None:
  optimizer_dict[parameter].step()
  optimizer_dict[parameter].zero_grad()

# Register the hook onto every parameter
for p in model.parameters():
   p.register_post_accumulate_grad_hook(optimizer_hook)

# Now remember our previous ``train()`` function? Since the optimizer has been
# fused into the backward, we can remove the optimizer step and zero_grad calls.
def train(model):
  # create our fake image input: tensor shape is batch_size, channels, height, width
  fake_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()

  # call our forward and backward
  loss = model.forward(fake_image)
  loss.sum().backward()

  # optimizer update --> no longer needed!
  # optimizer.step()
  # optimizer.zero_grad()

########################################################################
# That took about 10 lines of changes in our sample model, which is neat.
# However, for real models, it could be a fairly intrusive change to switch
# out the optimizer for an optimizer dictionary, especially for those who use
# ``LRScheduler``s or manipulate optimizer configuration throughout the
# training epochs. Working out this API with those changes will be more
# involved and will likely require moving more configuration into global
# state but should not be impossible. That said, a next step for PyTorch
# is to make this API easier to adopt with LRSchedulers and other features
# you are already used to.
# 
# But let me get back to convincing you that this technique is worth it.
# We will consult our friend, the memory snapshot.

# delete optimizer memory from before to get a clean slate for the next
# memory snapshot
del optimizer

# tell CUDA to start recording memory allocations
torch.cuda.memory._record_memory_history(enabled='all')

# train 3 steps. note that we no longer pass the optimizer into train()
for _ in range(3):
  train(model)

# save a snapshot of the memory allocations
s = torch.cuda.memory._snapshot()
with open(f"snapshot-opt-in-bwd.pickle", "wb") as f:
    dump(s, f)

# tell CUDA to stop recording memory allocations now
torch.cuda.memory._record_memory_history(enabled=None)

###############################################################################
# Yes, take some time to drag your snapshot into the CUDA Memory Visualizer.
# 
# .. figure:: /_static/img/optim_step_in_bwd/snapshot_opt_in_bwd.jpg
#    :alt: snapshot.png loaded into CUDA Memory Visualizer
#
# Several major observations:
#  1. There is no more optimizer step! Right...we fused that into the backward.
#  2. Likewise, the backward drags longer and there are more random allocations
#     for intermediates. This is expected, as the optimizer step requires 
#     intermediates.
#  3. Most importantly! The peak memory is lower! It is now ~4GB (which I
#     hope maps closely to your earlier expectation). 
# 
# Note that there is no longer any big chunk of memory allocated for the gradients
# compared to before, accounting for ~1.2GB of memory savings. Instead, we've freed
# each gradient very quickly after they've been computed by moving the optimizer 
# step as far ahead as we can. Woohoo! By the way, the other ~1.2GB of memory savings
# comes from breaking apart the optimizer into per-parameter optimizers, so the
# intermediates have proportionally shrunk. This detail is `less important` than
# the gradient memory savings, as you can get optimizer intermediates savings
# from just turning ``foreach=False`` without this technique.
# 
# You may be correctly wondering: if we saved 2.4GB of memory, why is the peak memory
# NOT 6GB - 2.4GB = 3.6GB? Well, the peak has moved! The peak is now near the start
# of the backward step, when we still have activations in memory, where before, the peak
# was during the optimizer step when the activations had been freed. The ~0.4GB difference
# accounting for ~4.0GB - ~3.6GB is thus due to the activations memory. One can then
# imagine that this technique can be coupled with activations checkpointing for more
# memory wins.
#
# Conclusion
# """"""""""
# In this tutorial, we learned about the memory saving technique of
# fusing the optimizer into the backward step through the new
# ``Tensor.register_post_accumulate_grad_hook()`` API and *when* to apply this
# technique (when gradients memory is significant). Along the way, we also learned
# about memory snapshots, which are generally useful in memory optimization.
