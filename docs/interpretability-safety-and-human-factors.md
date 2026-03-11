# Interpretability, Safety, and Human Factors

Extended reading list for model transparency, risk, fairness, and human-centered AI design.

Labels:

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Interpretability

- `[Free] [Intermediate] [Book]` [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/). Practical reference for feature attribution methods and their limits.
- `[Free] [Intermediate] [Paper]` [The Building Blocks of Interpretability (2018)](https://distill.pub/2018/building-blocks/). Strong visual introduction to neuron and feature-level interpretation.
- `[Free] [Intermediate] [Paper]` [Attention Is Not Explanation (2019)](https://arxiv.org/abs/1902.10186). Useful correction to common misuse of attention as explanation.
- `[Free] [Intermediate] [Paper]` [SHAP: A Unified Approach to Interpreting Model Predictions (2017)](https://arxiv.org/abs/1705.07874). Core paper for Shapley-value-based feature attribution.
- `[Free] [Intermediate] [Paper]` [Model Cards (2019)](https://arxiv.org/abs/1810.03993). Strong pattern for documenting intended use, limits, and evaluation context.
- `[Free] [Intermediate] [Paper]` [Datasheets for Datasets (2018)](https://arxiv.org/abs/1803.09010). Useful complement to model cards for data provenance and documentation.

## Safety and Alignment

- `[Free] [Intermediate] [Paper]` [Concrete Problems in AI Safety (2016)](https://arxiv.org/abs/1606.06565). Durable framing of reward hacking, safe exploration, and robustness issues.
- `[Free] [Intermediate] [Paper]` [Constitutional AI: Harmlessness from AI Feedback (2022)](https://arxiv.org/abs/2212.08073). Important for understanding AI-feedback-based alignment methods.
- `[Free] [Intermediate] [Book]` [Fairness and Machine Learning (2023)](https://fairmlbook.org/). Broad grounding in fairness concepts across technical and social contexts.
- `[Free] [Intermediate] [Survey]` [Bias and Fairness in Large Language Models: A Survey (2024)](https://direct.mit.edu/coli/article/50/3/1097/121961/Bias-and-Fairness-in-Large-Language-Models-A). Useful literature map for LLM-specific fairness issues.
- `[Free] [Intermediate] [Reference]` [Stanford HAI AI Index](https://aiindex.stanford.edu/). Good annual reference for governance, adoption, and ecosystem trends.

## Human Factors and System Design

- `[Free] [Intermediate] [Paper]` [Guidelines for Human-AI Interaction (2019)](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/). Still one of the most useful design papers for AI-assisted systems.
- `[Free] [Intermediate] [Reference]` [Google PAIR Guidebook](https://pair.withgoogle.com/). Practical human-centered design patterns for AI interfaces.
- `[Free] [Intermediate] [Paper]` [Human-Centered AI (2020)](https://arxiv.org/abs/2001.02805). Useful framing for high-control, human-supervised AI systems.
- `[Free] [Intermediate] [Reference]` [Microsoft AI UX Cookbook](https://github.com/microsoft/ai-ux-cookbook). Practical examples for explanation, uncertainty, and interaction patterns.

## Study Order

1. Start with `Interpretable Machine Learning` and `Guidelines for Human-AI Interaction`.
2. Read `Concrete Problems in AI Safety` once you want a durable taxonomy of failure modes.
3. Add `Fairness and Machine Learning` before making strong claims about bias mitigation.
4. Use `Model Cards` and `Datasheets for Datasets` as operating documents, not just reading material.
