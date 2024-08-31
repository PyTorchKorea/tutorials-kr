전체론적 추적 분석 소개
=======================================

**저자:** `Anupam Bhatnagar <https://github.com/anupambhatnagar>`_

이 튜토리얼에서는 분산 학습 작업의 추적을 분석하기 위해 전체론적 추적 분석(Holistic Trace Analysis, HTA)을 사용하는 방법을 보여줍니다. 시작하려면 아래 단계를 따르세요.

HTA 설치하기
~~~~~~~~~~~~~~

HTA를 설치하기 위해 Conda 환경을 사용하는 것을 권장합니다. Anaconda를 설치하려면 `공식 Anaconda 문서 <https://docs.anaconda.com/anaconda/install/index.html>`_를 참조하세요.

1. pip를 사용하여 HTA 설치:

   .. code-block:: python

      pip install HolisticTraceAnalysis

2. (선택사항이지만 권장) Conda 환경 설정:

   .. code-block:: python

      # env_name 환경 생성
      conda create -n env_name

      # 환경 활성화
      conda activate env_name

      # 작업이 끝나면 ``conda deactivate``를 실행하여 환경을 비활성화하세요

시작하기
~~~~~~~~~~~~~~~

Jupyter 노트북을 실행하고 `trace_dir` 변수를 추적 파일이 있는 위치로 설정하세요.

.. code-block:: python

   from hta.trace_analysis import TraceAnalysis
   trace_dir = "/path/to/folder/with/traces"
   analyzer = TraceAnalysis(trace_dir=trace_dir)


시간적 분석
------------------

GPU를 효과적으로 활용하기 위해서는 특정 작업에 대해 GPU가 시간을 어떻게 사용하고 있는지 이해하는 것이 중요합니다. GPU가 주로 계산, 통신, 메모리 이벤트에 사용되고 있는지, 아니면 유휴 상태인지? 시간적 분석 기능은 이 세 가지 범주에서 사용된 시간에 대한 상세한 분석을 제공합니다.

* 유휴 시간 - GPU가 유휴 상태입니다.
* 계산 시간 - GPU가 행렬 곱셈이나 벡터 연산에 사용되고 있습니다.
* 비계산 시간 - GPU가 통신이나 메모리 이벤트에 사용되고 있습니다.

높은 학습 효율성을 달성하기 위해서는 코드가 계산 시간을 최대화하고 유휴 시간과 비계산 시간을 최소화해야 합니다. 다음 함수는 각 랭크에 대한 시간 사용의 상세한 분석을 제공하는 데이터프레임을 생성합니다.

.. code-block:: python

   analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
   time_spent_df = analyzer.get_temporal_breakdown()


.. image:: ../_static/img/hta/temporal_breakdown_df.png

`get_temporal_breakdown <https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_temporal_breakdown>`_ 함수에서 `visualize` 인수를 `True`로 설정하면 랭크별 분석을 나타내는 막대 그래프도 생성됩니다.

.. image:: ../_static/img/hta/temporal_breakdown_plot.png


유휴 시간 분석
-------------------

GPU가 유휴 상태로 보내는 시간과 그 이유에 대한 통찰을 얻으면 최적화 전략을 수립하는 데 도움이 될 수 있습니다. GPU에서 실행 중인 커널이 없을 때 GPU는 유휴 상태로 간주됩니다. 우리는 `유휴` 시간을 세 가지 뚜렷한 범주로 분류하는 알고리즘을 개발했습니다:

* **호스트 대기:** CPU가 GPU를 완전히 활용하기 위해 충분히 빠르게 커널을 대기열에 넣지 않아 발생하는 GPU의 유휴 시간을 말합니다. 이러한 유형의 비효율성은 속도 저하에 기여하는 CPU 연산자를 검사하고, 배치 크기를 늘리고, 연산자 융합을 적용하여 해결할 수 있습니다.

* **커널 대기:** GPU에서 연속적인 커널을 실행하는 것과 관련된 간단한 오버헤드를 말합니다. 이 범주에 속하는 유휴 시간은 CUDA 그래프 최적화를 사용하여 최소화할 수 있습니다.

* **기타 대기:** 현재 정보가 부족하여 귀속시킬 수 없는 유휴 시간이 이 범주에 포함됩니다. 가능한 원인으로는 CUDA 이벤트를 사용한 CUDA 스트림 간의 동기화와 커널 실행 지연 등이 있습니다.

호스트 대기 시간은 CPU로 인해 GPU가 정체되는 시간으로 해석할 수 있습니다. 유휴 시간을 커널 대기로 귀속시키기 위해 우리는 다음과 같은 휴리스틱을 사용합니다:

   | **연속적인 커널 사이의 간격 < 임계값**

기본 임계값은 30 나노초이며 `consecutive_kernel_delay` 인수를 사용하여 구성할 수 있습니다. 기본적으로 유휴 시간 분석은 랭크 0에 대해서만 계산됩니다. 다른 랭크에 대해 분석을 계산하려면 `get_idle_time_breakdown <https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_idle_time_breakdown>`_ 함수에서 `ranks` 인수를 사용하세요. 유휴 시간 분석은 다음과 같이 생성할 수 있습니다:

.. code-block:: python

  analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
  idle_time_df = analyzer.get_idle_time_breakdown()

.. image:: ../_static/img/hta/idle_time_breakdown_percentage.png

이 함수는 데이터프레임 튜플을 반환합니다. 첫 번째 데이터프레임은 각 랭크의 각 스트림에 대한 유휴 시간 범주별 시간을 포함합니다.

.. image:: ../_static/img/hta/idle_time.png
   :scale: 100%
   :align: center

두 번째 데이터프레임은 `show_idle_interval_stats`가 `True`로 설정되었을 때 생성됩니다. 이 데이터프레임은 각 랭크의 각 스트림에 대한 유휴 시간의 요약 통계를 포함합니다.

.. image:: ../_static/img/hta/idle_time_summary.png
   :scale: 100%

.. tip::

   기본적으로 유휴 시간 분석은 각 유휴 시간 범주의 백분율을 표시합니다. `visualize_pctg` 인수를 `False`로 설정하면 함수는 y축에 절대 시간을 표시합니다.


커널 분석
----------------

커널 분석 기능은 모든 랭크에서 통신(COMM), 계산(COMP), 메모리(MEM)와 같은 각 커널 유형에 대해 사용된 시간을 분석하고 각 범주에서 사용된 시간의 비율을 제시합니다. 다음은 각 범주에서 사용된 시간의 백분율을 원형 차트로 나타낸 것입니다:

.. image:: ../_static/img/hta/kernel_type_breakdown.png
   :align: center

커널 분석은 다음과 같이 계산할 수 있습니다:

.. code-block:: python

   analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
   kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown()

함수가 반환하는 첫 번째 데이터프레임은 원형 차트를 생성하는 데 사용된 원래 값을 포함합니다.

Kernel Duration Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second dataframe returned by `get_gpu_kernel_breakdown
<https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_gpu_kernel_breakdown>`_
contains duration summary statistics for each kernel. In particular, this
includes the count, min, max, average, standard deviation, sum, and kernel type
for each kernel on each rank.

.. image:: ../_static/img/hta/kernel_metrics_df.png
   :align: center

Using this data HTA creates many visualizations to identify performance
bottlenecks.

#. Pie charts of the top kernels for each kernel type for each rank.

#. Bar graphs of the average duration across all ranks for each of the top
   kernels and for each kernel type.

.. image:: ../_static/img/hta/pie_charts.png

.. tip::

   All images are generated using plotly. Hovering on the graph shows the
   mode bar on the top right which allows the user to zoom, pan, select, and
   download the graph.

The pie charts above show the top 5 computation, communication, and memory
kernels. Similar pie charts are generated for each rank. The pie charts can be
configured to show the top k kernels using the ``num_kernels`` argument passed
to the `get_gpu_kernel_breakdown` function. Additionally, the
``duration_ratio`` argument can be used to tune the percentage of time that
needs to be analyzed. If both ``num_kernels`` and ``duration_ratio`` are
specified, then ``num_kernels`` takes precedence.

.. image:: ../_static/img/hta/comm_across_ranks.png

The bar graph above shows the average duration of the NCCL AllReduce kernel
across all the ranks. The black lines indicate the minimum and maximum time
taken on each rank.

.. warning::
   When using jupyter-lab set the "image_renderer" argument value to
   "jupyterlab" otherwise the graphs will not render in the notebook.

For a detailed walkthrough of this feature see the `gpu_kernel_breakdown
notebook
<https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/examples/kernel_breakdown_demo.ipynb>`_
in the examples folder of the repo.


Communication Computation Overlap
---------------------------------

In distributed training, a significant amount of time is spent in communication
and synchronization events between GPUs. To achieve high GPU efficiency (such as
TFLOPS/GPU), it is crucial to keep the GPU oversubscribed with computation
kernels. In other words, the GPU should not be blocked due to unresolved data
dependencies. One way to measure the extent to which computation is blocked by
data dependencies is to calculate the communication computation overlap. Higher
GPU efficiency is observed if communication events overlap computation events.
Lack of communication and computation overlap will lead to the GPU being idle,
resulting in low efficiency.
To sum up, a higher communication computation overlap is desirable. To calculate
the overlap percentage for each rank, we measure the following ratio:

  | **(time spent in computation while communicating) / (time spent in communication)**

The communication computation overlap can be calculated as follows:

.. code-block:: python

   analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
   overlap_df = analyzer.get_comm_comp_overlap()

The function returns a dataframe containing the overlap percentage
for each rank.

.. image:: ../_static/img/hta/overlap_df.png
   :align: center
   :scale: 50%

When the ``visualize`` argument is set to True, the `get_comm_comp_overlap
<https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_comm_comp_overlap>`_
function also generates a bar graph representing the overlap by rank.

.. image:: ../_static/img/hta/overlap_plot.png


Augmented Counters
------------------

Memory Bandwidth & Queue Length Counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Memory bandwidth counters measure the memory copy bandwidth used while copying
the data from H2D, D2H and D2D by memory copy (memcpy) and memory set (memset)
events. HTA also computes the number of outstanding operations on each CUDA
stream. We refer to this as **queue length**. When the queue length on a stream
is 1024 or larger new events cannot be scheduled on that stream and the CPU
will stall until the events on the GPU stream have processed.

The `generate_trace_with_counters
<https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.generate_trace_with_counters>`_
API outputs a new trace file with the memory bandwidth and queue length
counters. The new trace file contains tracks which indicate the memory
bandwidth used by memcpy/memset operations and tracks for the queue length on
each stream. By default, these counters are generated using the rank 0
trace file, and the new file contains the suffix ``_with_counters`` in its name.
Users have the option to generate the counters for multiple ranks by using the
``ranks`` argument in the ``generate_trace_with_counters`` API.

.. code-block:: python

  analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
  analyzer.generate_trace_with_counters()

A screenshot of the generated trace file with augmented counters.

.. image:: ../_static/img/hta/mem_bandwidth_queue_length.png
   :scale: 100%

HTA also provides a summary of the memory copy bandwidth and queue length
counters as well as the time series of the counters for the profiled portion of
the code using the following API:

* `get_memory_bw_summary <https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_memory_bw_summary>`_

* `get_queue_length_summary <https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_queue_length_summary>`_

* `get_memory_bw_time_series <https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_memory_bw_time_series>`_

* `get_queue_length_time_series <https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_queue_length_time_series>`_

To view the summary and time series, use:

.. code-block:: python

  # generate summary
  mem_bw_summary = analyzer.get_memory_bw_summary()
  queue_len_summary = analyzer.get_queue_length_summary()

  # get time series
  mem_bw_series = analyzer.get_memory_bw_time_series()
  queue_len_series = analyzer.get_queue_length_series()

The summary contains the count, min, max, mean, standard deviation, 25th, 50th,
and 75th percentile.

.. image:: ../_static/img/hta/queue_length_summary.png
   :scale: 100%
   :align: center

The time series only contains the points when a value changes. Once a value is
observed the time series stays constant until the next update. The memory
bandwidth and queue length time series functions return a dictionary whose key
is the rank and the value is the time series for that rank. By default, the
time series is computed for rank 0 only.

CUDA Kernel Launch Statistics
-----------------------------

.. image:: ../_static/img/hta/cuda_kernel_launch.png

For each event launched on the GPU, there is a corresponding scheduling event on
the CPU, such as ``CudaLaunchKernel``, ``CudaMemcpyAsync``, ``CudaMemsetAsync``.
These events are linked by a common correlation ID in the trace - see the figure
above. This feature computes the duration of the CPU runtime event, its corresponding GPU
kernel and the launch delay, for example, the difference between GPU kernel starting and
CPU operator ending. The kernel launch info can be generated as follows:

.. code-block:: python

  analyzer = TraceAnalysis(trace_dir="/path/to/trace/dir")
  kernel_info_df = analyzer.get_cuda_kernel_launch_stats()

A screenshot of the generated dataframe is given below.

.. image:: ../_static/img/hta/cuda_kernel_launch_stats.png
   :scale: 100%
   :align: center

The duration of the CPU op, GPU kernel, and the launch delay allow us to find
the following:

* **Short GPU kernels** - GPU kernels with duration less than the corresponding
  CPU runtime event.

* **Runtime event outliers** - CPU runtime events with excessive duration.

* **Launch delay outliers** - GPU kernels which take too long to be scheduled.

HTA generates distribution plots for each of the aforementioned three categories.

**Short GPU kernels**

Typically, the launch time on the CPU side ranges from 5-20 microseconds. In some
cases, the GPU execution time is lower than the launch time itself. The graph
below helps us to find how frequently such instances occur in the code.

.. image:: ../_static/img/hta/short_gpu_kernels.png


**Runtime event outliers**

The runtime outliers depend on the cutoff used to classify the outliers, hence
the `get_cuda_kernel_launch_stats
<https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_cuda_kernel_launch_stats>`_
API provides the ``runtime_cutoff`` argument to configure the value.

.. image:: ../_static/img/hta/runtime_outliers.png

**Launch delay outliers**

The launch delay outliers depend on the cutoff used to classify the outliers,
hence the `get_cuda_kernel_launch_stats` API provides the
``launch_delay_cutoff`` argument to configure the value.

.. image:: ../_static/img/hta/launch_delay_outliers.png


Conclusion
~~~~~~~~~~

In this tutorial, you have learned how to install and use HTA,
a performance tool that enables you analyze bottlenecks in your distributed
training workflows. To learn how you can use the HTA tool to perform trace
diff analysis, see `Trace Diff using Holistic Trace Analysis <https://tutorials.pytorch.kr/beginner/hta_trace_diff_tutorial.html>`__.
