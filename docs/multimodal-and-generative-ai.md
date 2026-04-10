# Multimodal and Generative AI

Reading list for vision-language models, image generation beyond the basics, and multimodal systems. Items in the main [README](../README.md) are not repeated here; start there for the essentials. See also the Generative Models section in the README for foundational diffusion and GAN papers.

Labels:

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Vision-Language Models

- `[Free] [Intermediate] [Paper]` [Learning Transferable Visual Models From Natural Language Supervision (CLIP, 2021)](https://arxiv.org/abs/2103.00020). Foundational paper for connecting images and text in a shared embedding space. Enables zero-shot image classification and underpins many later multimodal systems.
- `[Free] [Intermediate] [Paper]` [Visual Instruction Tuning (LLaVA, 2023)](https://arxiv.org/abs/2304.08485). Practical approach to building vision-language chatbots by combining a visual encoder with a language model.
- `[Free] [Intermediate] [Survey]` [A Survey on Multimodal Large Language Models (2024)](https://arxiv.org/abs/2306.13549). Broad overview of architectures, training methods, and benchmarks for multimodal LLMs.

## Image Generation

- `[Free] [Intermediate] [Paper]` [Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2, 2022)](https://arxiv.org/abs/2204.06125). Demonstrates how CLIP embeddings can guide high-quality text-to-image generation.
- `[Free] [Intermediate] [Paper]` [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen, 2022)](https://arxiv.org/abs/2205.11487). Shows the importance of strong text encoders for image generation quality.
- `[Free] [Advanced] [Paper]` [Scalable Diffusion Models with Transformers (DiT, 2023)](https://arxiv.org/abs/2212.09748). Replaces U-Net with transformer architecture in diffusion models. Relevant to understanding Sora and newer generators.

## Audio and Speech

- `[Free] [Intermediate] [Paper]` [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper, 2022)](https://arxiv.org/abs/2212.04356). Practical, high-quality speech recognition trained on diverse web audio.
- `[Free] [Intermediate] [Reference]` [Whisper (OpenAI)](https://github.com/openai/whisper). Reference implementation for speech-to-text with multilingual support.

## Study Order

1. Read `CLIP` first — it underpins most modern vision-language work.
2. Read `LLaVA` to understand how vision-language chatbots are built.
3. Read `DALL-E 2` or `Imagen` after the diffusion papers in the main README.
4. Use the multimodal LLM survey to map the broader landscape.
5. Explore `Whisper` if your work involves speech or audio.
