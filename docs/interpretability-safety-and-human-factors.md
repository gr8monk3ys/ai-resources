# Interpretability, Safety, and Human Factors

Extended reading for model transparency, risk, fairness, and human-centered AI design. Items in the main [README](../README.md) are not repeated here; start there for the essentials.

Labels:

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Interpretability

- `[Free] [Intermediate] [Paper]` [The Building Blocks of Interpretability (2018)](https://distill.pub/2018/building-blocks/). Strong visual introduction to neuron and feature-level interpretation.
- `[Free] [Intermediate] [Paper]` [Attention Is Not Explanation (2019)](https://arxiv.org/abs/1902.10186). Useful correction to common misuse of attention as explanation.
- `[Free] [Intermediate] [Paper]` [SHAP: A Unified Approach to Interpreting Model Predictions (2017)](https://arxiv.org/abs/1705.07874). Core paper for Shapley-value-based feature attribution.
- `[Free] [Intermediate] [Paper]` [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (2023)](https://transformer-circuits.pub/2023/monosemantic-features/). Landmark mechanistic interpretability work identifying individual features in neural networks.
- `[Free] [Intermediate] [Paper]` [Model Cards (2019)](https://arxiv.org/abs/1810.03993). Strong pattern for documenting intended use, limits, and evaluation context.
- `[Free] [Intermediate] [Paper]` [Datasheets for Datasets (2018)](https://arxiv.org/abs/1803.09010). Useful complement to model cards for data provenance and documentation.

## Safety and Alignment

- `[Free] [Intermediate] [Paper]` [Constitutional AI: Harmlessness from AI Feedback (2022)](https://arxiv.org/abs/2212.08073). Important for understanding AI-feedback-based alignment methods.
- `[Free] [Intermediate] [Survey]` [Bias and Fairness in Large Language Models: A Survey (2024)](https://arxiv.org/abs/2309.00770). Useful literature map for LLM-specific fairness issues.
- `[Free] [Intermediate] [Reference]` [Stanford HAI AI Index](https://hai.stanford.edu/ai-index). Good annual reference for governance, adoption, and ecosystem trends.
- `[Free] [Intermediate] [Reference]` [NIST AI Risk Management Framework (2023)](https://www.nist.gov/artificial-intelligence/executive-order-safe-secure-and-trustworthy-artificial-intelligence). Practical risk management guidance for organizations deploying AI systems.
- `[Free] [Intermediate] [Reference]` [EU AI Act Overview](https://artificialintelligenceact.eu/). Reference for understanding risk-based AI regulation. Relevant to anyone deploying AI in or for EU markets.

## Human Factors and System Design

- `[Free] [Intermediate] [Paper]` [Guidelines for Human-AI Interaction (2019)](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/). Still one of the most useful design papers for AI-assisted systems.
- `[Free] [Intermediate] [Reference]` [Google PAIR Guidebook](https://pair.withgoogle.com/). Practical human-centered design patterns for AI interfaces.
- `[Free] [Intermediate] [Paper]` [Human-Centered AI (2020)](https://arxiv.org/abs/2001.02805). Useful framing for high-control, human-supervised AI systems.
- `[Free] [Intermediate] [Reference]` [Microsoft HAX Toolkit](https://www.microsoft.com/en-us/research/project/hax-toolkit/). Practical guidance and examples for explanation, uncertainty, and interaction patterns.

## Study Order

1. Start with `Interpretable Machine Learning` and `Concrete Problems in AI Safety` (both in README), then `Guidelines for Human-AI Interaction` here.
2. Add `Fairness and Machine Learning` (in README) before making strong claims about bias mitigation.
3. Use `Model Cards` and `Datasheets for Datasets` as operating documents, not just reading material.
4. Read `Constitutional AI` once you want to understand AI-feedback-based alignment.
