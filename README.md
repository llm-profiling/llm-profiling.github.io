![deployment](https://img.shields.io/github/deployments/llm-profiling/llm-profiling.github.io/github-pages?label=deployment) ![Platform](https://img.shields.io/badge/platform-Linux-8A2BE2) ![license](https://img.shields.io/github/license/llm-profiling/llm-profiling.github.io)

## 0. Table of Contents

* [Introduction](#1-introduction)
* [Code](#2-code)
* [Usage](#3-usage)
* [Case](#4-case)

## 1. Introduction

Training and serving large language models (LLMs) has become a core business for AI providers. To ensure high-quality user experience while optimizing infrastructure costs, providers need to closely monitor the performance of LLM executions in production. However, existing performance profiling tools fall short in this context, which are either too coarse-grained to capture performance bottlenecks, or too intrusive that incur significant performance degradations.

In this paper, we present LMTracer, a fine-grained and real-time performance profiling framework designed for LLM services in production. We find that the main reason for existing techniques to be inefficient comes from the disruption in highly asynchronous GPU execution flows. To address this, our key idea is to embed profiling logic into the execution through graph-embedded fencing, which compiles lightweight fence kernels that realize the profiling logic as special trace nodes in the GPU execution graph, and streams buffered profiling data to CPUs on demand for keeping the execution of user kernels uninterrupted. LMTracer has been deployed in production for over four months. It incurs only 0.58% overhead on average across various LLM workloads (including training, serving, and fine-tuning), and has proactively uncovered 15 latent performance issues that would otherwise translate into severe system failures. After fixing these issues, we have observed 21.2% latency reduction and 24.5% throughput improvement in LLM services.

## 2. Code

The code for LMTracer is available [here](https://github.com/llm-profiling/llm-profiling.github.io/tree/main/lmtracer).

The directory is organized as follows.

<table>
  <tr>
    <td>Directory</td>
    <td>Description</td>
    <td>Source Code</td>
  </tr>
  <tr>
    <td rowspan='6'>csrc</td>
    <td rowspan='6'>Fence kernel implementations that read GPU clock registers and write the data to the shadow buffers. When one of the shadow buffers is full, fence kernels issue doorbell signals to CPU asynchronously through CPU-GPU mapped memory.</td>
    <td>globaltimer.cu</td>
  </tr>
  <tr><td >layout.h</td></tr>
  <tr><td >timer_kernel.cu</td></tr>
  <tr><td >timer_kernel.cuh</td></tr>
  <tr><td >timer_mem.cu</td></tr>
  <tr><td >timer_op.cu</td></tr>
  <tr>
    <td rowspan='3'>examples</td>
    <td rowspan='3'>End-to-end benchmark and profiling presets.</td>
    <td>deepseek_sglang.yaml</td>
  </tr>
  <tr><td>qwen_vllm.yaml</td></tr>
  <tr><td>multi_exporter_config.yaml</td></tr>
  <tr>
    <td rowspan='10'>lmtracer/core</td>
    <td rowspan='10'>Core runtime of LMTracer: probe configuration/context management, runtime hooks, shadow buffer readout, parsing, metrics, and exporter abstractions.</td>
    <td>probe_context.py</td>
  </tr>
  <tr><td>probe_config.py</td></tr>
  <tr><td>decorators.py</td></tr>
  <tr><td>hooks.py</td></tr>
  <tr><td>process_binder.py</td></tr>
  <tr><td>trace_buffer.py</td></tr>
  <tr><td>dump_parser.py</td></tr>
  <tr><td>metrics.py</td></tr>
  <tr><td>exporter.py</td></tr>
  <tr><td>time_sync.py</td></tr>
  <tr>
    <td rowspan='3'>lmtracer/cuda</td>
    <td rowspan='3'>Python bindings/type stubs for CUDA extensions used by probe recording and clock reads.</td>
    <td>globaltimer.pyi</td>
  </tr>
  <tr><td>timer_kernel.pyi</td></tr>
  <tr><td>timer_mem.pyi</td></tr>
  <tr>
    <td rowspan='8'>lmtracer/plugins</td>
    <td rowspan='8'>Framework integration layer. Patches or wraps upstream runtime entrypoints so LMTracer can initialize and flush traces during serving/training.</td>
    <td>vllm/platform.py</td>
  </tr>
  <tr><td>vllm/v1/worker/lmtracer_gpu_worker.py</td></tr>
  <tr><td>vllm/v1/worker/lmtracer_gpu_model_runner.py</td></tr>
  <tr><td>sglang/srt/model_executor/model_runner.py</td></tr>
  <tr><td>megatron/training.py</td></tr>
  <tr><td>torch/_inductor/scheduler.py</td></tr>
  <tr><td>torch/_inductor/wrapper.py</td></tr>
  <tr><td>torch/_inductor/cuda_combined_scheduling.py</td></tr>
  <tr>
    <td rowspan='2'>tools</td>
    <td rowspan='2'>Utilities for distributed command execution and converting binary dumps into Perfetto traces.</td>
    <td>dump_parser.py</td>
  </tr>
  <tr><td>dump_parser_multitrack.py</td></tr>
  <tr>
    <td rowspan='2'>tests</td>
    <td rowspan='2'>Benchmark scripts and sanity checks for OpenAI-compatible endpoints and Megatron training workloads.</td>
    <td>run_benchmark.py</td>
  </tr>
  <tr><td>openai_perf.py</td></tr>
  
</table>

## 3. Usage

After cloning the repository, you can install LMTracer with the following command:

```bash
pip install ./lmtracer --no-build-isolation
```

> Note: LMTracer does not support editable mode (i.e., the *-e* option for *pip install*) of Python installation, or the hooks to LLM frameworks will be invalidated.

Then, LMTracer can be automatically enabled for supported frameworks (e.g., vLLM, SGLang, and Megatron) and models (e.g., DeepSeek V3, Qwen3, and GPT OSS) without any code changes.

The supported versions of the tested LLM frameworks are:

* **vLLM** v0.9.2, v0.11.2
* **SGLang** v0.5.6, v0.5.7
* **MegatronLM** v0.15.2

You can also specify custom profiling configurations to capture different levels of details for different frameworks and models.

An example of LMTracer configuration is as follows.

```yaml
llm_engine: sglang
probes:
  # LLM modules
  - phase: model
    module: sglang.srt.models.gpt_oss.GptOssModel
  - phase: decoder_layer
    module: sglang.srt.models.gpt_oss.GptOssDecoderLayer
  - phase: mlp
    module: sglang.srt.models.gpt_oss.GptOssSparseMoeBlock
  - phase: attention
    module: sglang.srt.models.gpt_oss.GptOssAttention
  # Functions or even code lines
  - phase: norm
    function: sglang.srt.layers.layernorm.RMSNorm.forward_cuda
    # Optional: Flexibility for code-line profiling
    line_start: 109
    line_end: 126
exporters:
  # Exporter 1: Save binary trace data to files
  - module: lmtracer.core.exporter.FileByteDataExporter
    params:
      output_dir: /tmp/lmtracer_exports
  # Exporter 2: Export metrics to Prometheus
  - module: lmtracer.core.exporter.SGLangMetricPrometheusExporter
    params: {}
```

An example profiling output for LLM inference is as follows.

<img width="1614" alt="example_timeline" src="https://github.com/user-attachments/assets/482d216a-a6c9-43f4-b2dd-533e1571a864" />

## 4. Case

After the real-world deployment, LMTracer has proactively uncovered 15 latent performance issues.
Below we list representative issues that LMTracer has helped identify, and the corresponding subsequent failures that would occur if the issue is not handled immediately.

<table>
  <tr>
    <td>Component</td>
    <td>Root Cause</td>
    <td>Symptom</td>
    <td>Subsequent Failure</td>
  </tr>
  <tr>
    <td>Configuration</td>
    <td>Improper layer placement</td>
    <td>PP stages are much longer.</td>
    <td>Users' SLA is not satisfied.</td>
  </tr>
  
  <tr>
    <td>Configuration</td>
    <td>Improper parallelism</td>
    <td>TP groups use slow paths.</td>
    <td>Users' SLA is not satisfied.</td>
  </tr>
  
  <tr>
    <td>Configuration</td>
    <td>Not using CUDA graph</td>
    <td>Separate kernel launching.</td>
    <td>Some GPUs straggle.</td>
  </tr>
  
  <tr>
    <td>Framework</td>
    <td>Unbalanced scheduling</td>
    <td>Some GPUs are busier.</td>
    <td>Inefficiency propagates.</td>
  </tr>
  
  <tr>
    <td>Framework</td>
    <td>KV cache eviction</td>
    <td>Repeated computation.</td>
    <td>Some GPUs straggle.</td>
  </tr>
  
  <tr>
    <td>Framework</td>
    <td>Prefill blocking</td>
    <td>High TTFT for users.</td>
    <td>Decoding GPUs are underutilized.</td>
  </tr>

  <tr>
    <td>Hardware</td>
    <td>NVLink down</td>
    <td>Training process is stalled.</td>
    <td>Training is silently halted.</td>
  </tr>

  <tr>
    <td>Hardware</td>
    <td>RNIC flapping</td>
    <td>Intermittent unconnectivity.</td>
    <td>Training tasks are failed.</td>
  </tr>
  
</table>
