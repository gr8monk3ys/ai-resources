# Evaluation and Deployment

Reading list for benchmarks, model evaluation, serving infrastructure, and production ML workflows. Items in the main [README](../README.md) are not repeated here; start there for the essentials.

Labels:

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Benchmarks and Evaluation

- `[Free] [Intermediate] [Paper]` [Measuring Massive Multitask Language Understanding (MMLU, 2020)](https://arxiv.org/abs/2009.03300). Widely used benchmark for evaluating broad knowledge across 57 subjects. Good context for reading model comparison papers.
- `[Free] [Intermediate] [Paper]` [Holistic Evaluation of Language Models (HELM, 2022)](https://arxiv.org/abs/2211.09110). Multi-metric evaluation framework that goes beyond accuracy to measure calibration, robustness, and fairness.
- `[Free] [Intermediate] [Reference]` [lm-evaluation-harness (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness). Standard open-source framework for running LLM benchmarks. Used by most open-weight model releases.
- `[Free] [Intermediate] [Reference]` [Chatbot Arena (LMSYS)](https://chat.lmsys.org/). Crowdsourced Elo rating system for comparing LLM quality through blind pairwise evaluation.

## Inference and Serving

- `[Free] [Intermediate] [Reference]` [vLLM](https://github.com/vllm-project/vllm). High-throughput serving engine using paged attention. Good default choice for production LLM inference.
- `[Free] [Intermediate] [Reference]` [llama.cpp](https://github.com/ggml-org/llama.cpp). Efficient CPU and GPU inference for quantized models. Useful for local deployment and resource-constrained environments.
- `[Free] [Intermediate] [Reference]` [Ollama](https://ollama.com/). Simplified local model running built on llama.cpp. Good for prototyping and development.
- `[Free] [Intermediate] [Reference]` [SGLang](https://github.com/sgl-project/sglang). Serving framework with structured generation support and efficient batching.

## Experiment Tracking and Monitoring

- `[Free] [Intermediate] [Reference]` [Weights & Biases](https://wandb.ai/). Experiment tracking, model versioning, and training visualization. Free tier available for individual use.
- `[Free] [Intermediate] [Reference]` [MLflow](https://mlflow.org/). Open-source platform for experiment tracking, model registry, and deployment. Good option if you prefer self-hosted tooling.

## Study Order

1. Read `MMLU` and `HELM` to understand how models are evaluated and why single-number comparisons are misleading.
2. Try `lm-evaluation-harness` or `Chatbot Arena` to ground your intuition in actual evaluation.
3. Pick a serving stack (`vLLM` for production scale, `Ollama` for local development) based on your deployment needs.
4. Set up experiment tracking early — it pays off as soon as you have more than a few training runs to compare.
