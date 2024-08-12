"""

옵티마이저 단계를 역전파 과정에 합쳐서 메모리 절약하기
======================================================================

안녕하세요! 이 튜토리얼에서는 *변화도(gradient)* 가 차지하는 메모리를 줄임으로써 
학습 단계(training loop)에서의 메모리 사용량을 줄이는 한 가지 방법을 소개합니다.
모델이 있는 상황에서 메모리 부족(Out of Memory, OOM) 오류를 방지하고 싶거나,
GPU의 성능을 최대한 활용하고 싶은 경우 이 방법이 도움이 될 수 있습니다.
(변화도가 메모리의 일부(partition)를 차지하고 있으며, 변화도 누적(accumulation)이 필요하지 않은 경우라면 말입니다.)

아래 내용을 다룹니다:

1. 학습 또는 미세조정(finetuning) 단계 중 메모리를 차지하는 요소들,
2. 메모리 스냅숏(snapshot)을 캡처하고 시각화하여 병목 현상을 파악하는 방법,
3. 새로운 ``Tensor.register_post_accumulate_grad_hook(hook)`` API, 그리고
4. 이 모든 것을 감안한 단 10줄의 코드로 메모리를 절약하는 법.

이 튜토리얼을 실행하기 위해 필요한 것:

*  2.1.0 혹은 그 이상의 버전의 PyTorch와 ``torchvision``
*  메모리 시각화를 로컬에서 실행하려면 CUDA GPU 1개
   메모리 시각화를 제외하면 이 방법은 모든 장치에서 유사한 이점을 제공합니다.

먼저 필요한 모듈과 모델을 import 하겠습니다. 
여기에서는 torchvision의 비전 트랜스포머 모델을 사용하지만, 다른 모델로 대체해도 좋습니다.
또 옵티마이저로 ``torch.optim.Adam`` 을 사용하지만, 마찬가지로 다른 옵티마이저로 대체해도 됩니다.


"""

import torch
from torchvision import models
from pickle import dump

model = models.vit_l_16(weights='DEFAULT').cuda()
optimizer = torch.optim.Adam(model.parameters())

###############################################################################
# Now let's define our typical training loop. You should use real images when
# training, but for the purposes of this tutorial, we are passing in fake
# inputs and not worrying about loading any actual data.

IMAGE_SIZE = 224

def train(model, optimizer):
  # create our fake image input: tensor shape is batch_size, channels, height, width
  fake_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()

  # call our forward and backward
  loss = model.forward(fake_image)
  loss.sum().backward()

  # optimizer update
  optimizer.step()
  optimizer.zero_grad()

###############################################################################
# Memory usage during training
# """"""""""""""""""""""""""""
# We are about to look at some memory snapshots, so we should be prepared to
# analyze them properly. Typically, training memory consists of:
#
#  * Model parameters (size P)
#  * Activations that are saved for the backward pass (size A)
#  * Gradients, which are the same size as the model parameters, so size G = P.
#  * Optimizer state, which is proportional to the size of the parameters. In
#    this case, the state for Adam requires 2x the model parameters, so size O = 2P.
#  * Intermediate tensors, which are allocated throughout the compute. We will
#    not worry about them for now as they are usually small and ephemeral.
#
# Capturing and visualizing memory snapshots
# """"""""""""""""""""""""""""""""""""""""""
# Let's get us a memory snapshot! As your code runs, consider what you may expect
# the CUDA memory timeline to look like.

# tell CUDA to start recording memory allocations
torch.cuda.memory._record_memory_history(enabled='all')

# train 3 steps
for _ in range(3):
  train(model, optimizer)

# save a snapshot of the memory allocations
s = torch.cuda.memory._snapshot()
with open(f"snapshot.pickle", "wb") as f:
    dump(s, f)

# tell CUDA to stop recording memory allocations now
torch.cuda.memory._record_memory_history(enabled=None)

###############################################################################
# Now open up the snapshot in the CUDA Memory Visualizer at
# https://pytorch.org/memory_viz by dragging and dropping the
# ``snapshot.pickle`` file. Does the memory timeline match your expectations?
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
# Where is the memory bottleneck in this training loop? Or, in other words,
# where is the peak memory?
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
# DISCLAIMER: This technique is **not** for all
# """""""""""""""""""""""""""""""""""""""""""""
# Before we get too excited, we have to consider whether this technique is applicable
# for `your` use case. This is NOT a silver bullet! The technique of fusing the 
# optimizer step into the backward only targets reducing *gradient* memory (and as a side effect also optimizer intermediates
# memory). Thus, the more sizable the memory taken up by the gradients, the more
# tantamount the memory reduction. In our example above, the gradients eat up 20% 
# of the memory pie, which is quite sizable!
#
# This may not be the case for you, for example, if your weights are already tiny,
# (say, due to applying LoRa,) then the gradients do not take much space in your
# training loop and the wins are way less exciting. In that case, you should
# first try other techniques like activations checkpointing, distributed
# training, quantization, or reducing the batch size. Then, when the gradients
# are part of the bottleneck again, come back to this tutorial!
# 
# Still here? Cool, let's introduce our new ``register_post_accumulate_grad_hook(hook)``
# API on Tensor.
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
