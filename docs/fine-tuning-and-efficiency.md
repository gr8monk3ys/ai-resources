# Fine-Tuning and Efficiency

Reading list for parameter-efficient training, quantization, scaling laws, and practical fine-tuning workflows. Items in the main [README](../README.md) are not repeated here; start there for the essentials.

Labels:

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Scaling Laws

- `[Free] [Intermediate] [Paper]` [Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361). Empirical study of how loss scales with model size, data, and compute. Essential context for understanding training budgets.
- `[Free] [Intermediate] [Paper]` [Training Compute-Optimal Large Language Models (Chinchilla, 2022)](https://arxiv.org/abs/2203.15556). Showed that most large models were undertrained relative to their size, reshaping how training budgets are allocated.

## Parameter-Efficient Fine-Tuning

- `[Free] [Intermediate] [Paper]` [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685). The most widely adopted PEFT method. Good starting point for understanding adapter-based fine-tuning.
- `[Free] [Intermediate] [Paper]` [QLoRA: Efficient Finetuning of Quantized Language Models (2023)](https://arxiv.org/abs/2305.14314). Combines quantization with LoRA to fine-tune large models on consumer hardware.
- `[Free] [Intermediate] [Survey]` [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning (2023)](https://arxiv.org/abs/2303.15647). Useful survey covering adapters, prefix tuning, prompt tuning, and LoRA variants in one place.
- `[Free] [Intermediate] [Reference]` [Hugging Face PEFT](https://huggingface.co/docs/peft/index). Practical library documentation for applying LoRA, prefix tuning, and other PEFT methods.

## Quantization and Compression

- `[Free] [Advanced] [Paper]` [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (2022)](https://arxiv.org/abs/2210.17323). Key paper for one-shot weight quantization to 4-bit and below.
- `[Free] [Advanced] [Paper]` [AWQ: Activation-aware Weight Quantization (2023)](https://arxiv.org/abs/2306.00978). Improved quantization that preserves salient weights based on activation patterns.
- `[Free] [Intermediate] [Paper]` [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)](https://arxiv.org/abs/2208.07339). Practical approach to 8-bit inference that handles outlier features.

## Practical Fine-Tuning Tools

- `[Free] [Intermediate] [Reference]` [TRL: Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index). Library for SFT, reward modeling, and preference optimization. Good entry point for hands-on RLHF/DPO work.
- `[Free] [Intermediate] [Reference]` [Axolotl](https://github.com/axolotl-ai-cloud/axolotl). Streamlined fine-tuning framework with good defaults for LoRA, QLoRA, and full fine-tuning.

## Study Order

1. Read the two scaling laws papers to understand why training decisions matter at scale.
2. Read `LoRA`, then `QLoRA` for the most practical fine-tuning path.
3. Use the PEFT survey to understand the broader landscape of adapter methods.
4. Move to quantization papers only once you need to deploy or serve models efficiently.
5. Pick up TRL or Axolotl when you are ready to fine-tune your own models.
