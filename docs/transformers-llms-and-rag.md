# Transformers, LLMs, and RAG

Extended reading list for modern language modeling, retrieval, and agent-style systems.

Labels:

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Core Transformer and LLM Papers

- `[Free] [Intermediate] [Paper]` [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762). Starting point for transformer architectures.
- `[Free] [Intermediate] [Paper]` [BERT: Pre-training of Deep Bidirectional Transformers (2018)](https://arxiv.org/abs/1810.04805). Important for encoder-style pretraining and fine-tuning.
- `[Free] [Intermediate] [Paper]` [Language Models are Few-Shot Learners (2020)](https://arxiv.org/abs/2005.14165). Useful historical reference for scaling and in-context learning.
- `[Free] [Intermediate] [Paper]` [Training Language Models to Follow Instructions with Human Feedback (2022)](https://arxiv.org/abs/2203.02155). Core paper for RLHF-style instruction tuning.
- `[Free] [Intermediate] [Paper]` [Direct Preference Optimization (2023)](https://arxiv.org/abs/2305.18290). Preference optimization without a separate reward model.

## Surveys and Overviews

- `[Free] [Intermediate] [Survey]` [A Survey of Large Language Models (2023)](https://arxiv.org/abs/2303.18223). Broad orientation to the LLM literature.
- `[Free] [Intermediate] [Survey]` [Foundations of Large Language Models (2023)](https://arxiv.org/abs/2307.09394). Useful companion survey focused on pretraining, scaling, and adaptation.
- `[Free] [Intermediate] [Survey]` [Efficient Large Language Models: A Survey (2024)](https://arxiv.org/abs/2312.03863). Good overview of quantization, pruning, and efficiency work.
- `[Free] [Intermediate] [Survey]` [RAG for LLMs: A Survey (2023)](https://arxiv.org/abs/2312.10997). Clear overview of naive, advanced, and modular retrieval pipelines.
- `[Free] [Advanced] [Survey]` [A Survey on LLM-based Autonomous Agents (2023)](https://arxiv.org/abs/2308.11432). Useful map of agent architectures, memory, planning, and tool use.

## Retrieval and Agent Systems

- `[Free] [Intermediate] [Reference]` [LLM Powered Autonomous Agents (2023)](https://lilianweng.github.io/posts/2023-06-23-agent/). Practical guide to common agent design patterns.
- `[Free] [Intermediate] [Reference]` [GraphRAG (Microsoft Research, 2024)](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/). Concrete example of graph-structured retrieval over private corpora.
- `[Free] [Advanced] [Survey]` [Unifying Large Language Models and Knowledge Graphs: A Roadmap (2024)](https://arxiv.org/abs/2306.08302). Useful if you want deeper grounding and reasoning support than plain vector retrieval provides.

## Implementation-Oriented References

- `[Free] [Intermediate] [Reference]` [Hugging Face Transformers](https://huggingface.co/docs/transformers/index). Practical documentation for loading, fine-tuning, and evaluating transformer models.
- `[Free] [Intermediate] [Reference]` [vLLM](https://github.com/vllm-project/vllm). Useful serving stack for efficient LLM inference.
- `[Free] [Intermediate] [Reference]` [llama.cpp](https://github.com/ggerganov/llama.cpp). Good reference point for local inference and constrained deployment environments.

## Study Order

1. Read `Attention Is All You Need`, `BERT`, and `Language Models are Few-Shot Learners` in that order.
2. Use one or two surveys to build the landscape before reading more implementation papers.
3. Read `RAG for LLMs` before choosing a retrieval architecture.
4. Treat agent frameworks as implementation tools, not substitutes for understanding planning, memory, and evaluation.
