# Transformers, LLMs, and RAG

Extended reading for modern language modeling, retrieval, and agent-style systems. Items in the main [README](../README.md) are not repeated here; start there for the essentials.

Labels:

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Core Transformer and LLM Papers

- `[Free] [Intermediate] [Paper]` [BERT: Pre-training of Deep Bidirectional Transformers (2018)](https://arxiv.org/abs/1810.04805). Important for encoder-style pretraining and fine-tuning.
- `[Free] [Intermediate] [Paper]` [Training Language Models to Follow Instructions with Human Feedback (2022)](https://arxiv.org/abs/2203.02155). Core paper for RLHF-style instruction tuning.

## Open-Weight Models

- `[Free] [Intermediate] [Paper]` [Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)](https://arxiv.org/abs/2307.09288). Important reference for understanding open-weight model training, RLHF application, and safety tuning at scale.
- `[Free] [Intermediate] [Paper]` [The Llama 3 Herd of Models (2024)](https://arxiv.org/abs/2407.21783). Details the training of Llama 3 across multiple sizes with multilingual and multimodal capabilities.
- `[Free] [Intermediate] [Paper]` [Mixtral of Experts (2024)](https://arxiv.org/abs/2401.04088). Demonstrates sparse mixture-of-experts at a practical scale. Important for understanding efficiency-quality tradeoffs in modern architectures.
- `[Free] [Intermediate] [Paper]` [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2024)](https://arxiv.org/abs/2405.04434). Notable for multi-head latent attention and economical MoE training.

## Reasoning and Inference-Time Compute

- `[Free] [Intermediate] [Paper]` [Let's Verify Step by Step (2023)](https://arxiv.org/abs/2305.20050). Shows that process-based reward models (verifying each reasoning step) outperform outcome-based approaches.
- `[Free] [Intermediate] [Paper]` [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (2024)](https://arxiv.org/abs/2407.21787). Demonstrates that scaling inference-time compute through repeated sampling can substantially improve problem-solving.

## Surveys and Overviews

- `[Free] [Intermediate] [Survey]` [Foundations of Large Language Models (2023)](https://arxiv.org/abs/2307.09394). Useful companion survey focused on pretraining, scaling, and adaptation.
- `[Free] [Intermediate] [Survey]` [Efficient Large Language Models: A Survey (2024)](https://arxiv.org/abs/2312.03863). Good overview of quantization, pruning, and efficiency work.
## Retrieval Systems

- `[Free] [Intermediate] [Reference]` [GraphRAG (Microsoft Research, 2024)](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/). Concrete example of graph-structured retrieval over private corpora.
- `[Free] [Advanced] [Survey]` [Unifying Large Language Models and Knowledge Graphs: A Roadmap (2024)](https://arxiv.org/abs/2306.08302). Useful if you want deeper grounding and reasoning support than plain vector retrieval provides.

> For agent architectures and tool use, see [Prompt Engineering and Agents](prompt-engineering-and-agents.md).

## Implementation-Oriented References

- `[Free] [Intermediate] [Reference]` [Hugging Face Transformers](https://huggingface.co/docs/transformers/index). Practical documentation for loading, fine-tuning, and evaluating transformer models.

> For inference serving and deployment tools, see [Evaluation and Deployment](evaluation-and-deployment.md).

## Study Order

1. Read `Attention Is All You Need` and `Language Models are Few-Shot Learners` (both in README), then `BERT` here.
2. Use one or two surveys to build the landscape before reading more implementation papers.
3. Read `RAG for LLMs` (in README) before choosing a retrieval architecture.
4. Treat agent frameworks as implementation tools, not substitutes for understanding planning, memory, and evaluation.
