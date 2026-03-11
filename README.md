# AI Learning Resources

Curated books, papers, courses, and reference material for learning artificial intelligence and machine learning.

This repository is selective by design. It is not a news feed, a leaderboard, or an exhaustive directory. The goal is to keep a short list of resources that remain useful after the hype cycle passes.

## Curation Principles

- Prefer primary sources, official course pages, and canonical books.
- Prefer free or open-access material when possible.
- Keep the main list short enough to skim in one sitting.
- Avoid hype, rankings, and fast-aging product details.
- Add only resources that are still likely to be worth recommending a year from now.

## Labels

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Start Here

| Goal | Suggested order |
| --- | --- |
| New to AI and ML | `AI: Foundations of Computational Agents` -> `Introduction to Statistical Learning` -> `Neural Networks and Deep Learning` -> `Dive into Deep Learning` |
| Want to build models | `Introduction to Statistical Learning` -> `Dive into Deep Learning` -> `fast.ai` -> `Machine Learning Yearning` |
| Focused on LLMs | `Attention Is All You Need` -> `BERT` -> `Language Models are Few-Shot Learners` -> `A Survey of Large Language Models` -> `RAG for LLMs: A Survey` |
| Interested in safety and interpretability | `Interpretable Machine Learning` -> `Attention Is Not Explanation` -> `Concrete Problems in AI Safety` -> `Fairness and Machine Learning` -> `Guidelines for Human-AI Interaction` |

## Prerequisites

- `[Free] [Beginner] [Course]` [Python for Everybody](https://www.py4e.com/). A good starting point if you are new to programming.
- `[Free] [Beginner] [Book]` [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/). Practical Python for people who learn best by building small scripts.
- `[Free] [Beginner] [Course]` [Khan Academy](https://www.khanacademy.org/). Use it to fill gaps in probability, statistics, linear algebra, and calculus.
- `[Free] [Beginner] [Course]` [3Blue1Brown Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr). Clear visual introduction to vectors, matrices, and transformations.

## Foundations

- `[Free] [Beginner] [Book]` [Artificial Intelligence: Foundations of Computational Agents (3rd Ed., 2023)](https://artint.info/). Broad introduction to search, reasoning, planning, uncertainty, agents, and societal implications.
- `[Free] [Beginner] [Book]` [Introduction to Statistical Learning (2nd Ed., 2021)](https://www.statlearning.com/). Strong starting point for practical machine learning with enough math to be useful.
- `[Free] [Beginner] [Book]` [The Hundred-Page Machine Learning Book (2019)](http://themlbook.com/wiki/doku.php). A compact orientation pass before deeper study.
- `[Free] [Beginner] [Reference]` [Machine Learning Yearning (2018)](https://www.mlyearning.org/). Especially useful once you start building projects and need help making modeling decisions.
- `[Free] [Advanced] [Reference]` [Probabilistic Machine Learning: Advanced Topics (2023)](https://probml.github.io/pml-book/book2.html). Strong reference for modern probabilistic methods after you have the basics.

## Deep Learning

- `[Free] [Beginner] [Book]` [Neural Networks and Deep Learning (2015)](http://neuralnetworksanddeeplearning.com/). Still one of the clearest introductions to backpropagation and neural network intuition.
- `[Free] [Intermediate] [Book]` [Deep Learning (2016)](https://www.deeplearningbook.org/). Classic theory-oriented reference for optimization, regularization, sequence models, and representation learning.
- `[Free] [Intermediate] [Book]` [Dive into Deep Learning (2023)](https://d2l.ai/). Hands-on and current enough to bridge fundamentals with implementation.
- `[Free] [Intermediate] [Book]` [Understanding Deep Learning (2023)](https://udlbook.github.io/udlbook/). Good modern companion once you want more depth on contemporary architectures.
- `[Free] [Intermediate] [Course]` [Deep Learning for Coders with fastai and PyTorch](https://course.fast.ai/). Useful if you learn better by training models early and filling in theory as needed.

## Transformers, LLMs, and Retrieval

- `[Free] [Intermediate] [Paper]` [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762). The paper that made transformers the default architecture for modern language models.
- `[Free] [Intermediate] [Paper]` [BERT: Pre-training of Deep Bidirectional Transformers (2018)](https://arxiv.org/abs/1810.04805). Important for understanding encoder-style pretraining and transfer learning.
- `[Free] [Intermediate] [Paper]` [Language Models are Few-Shot Learners (2020)](https://arxiv.org/abs/2005.14165). Helpful historical context for in-context learning and large-scale language modeling.
- `[Free] [Intermediate] [Survey]` [A Survey of Large Language Models (2023)](https://arxiv.org/abs/2303.18223). Widely used survey for orienting yourself in the LLM literature.
- `[Free] [Intermediate] [Survey]` [RAG for LLMs: A Survey (2023)](https://arxiv.org/abs/2312.10997). Good overview of retrieval-augmented generation patterns and tradeoffs.
- `[Free] [Intermediate] [Reference]` [LLM Powered Autonomous Agents (2023)](https://lilianweng.github.io/posts/2023-06-23-agent/). A practical systems-oriented guide to memory, planning, and tool use.

## Generative Models

- `[Free] [Intermediate] [Paper]` [Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661). The starting point for modern adversarial generative modeling.
- `[Free] [Intermediate] [Paper]` [Auto-Encoding Variational Bayes (2013)](https://arxiv.org/abs/1312.6114). Core reading for latent variable models and variational inference.
- `[Free] [Intermediate] [Paper]` [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239). The key diffusion paper to read before newer image and video model variants.
- `[Free] [Intermediate] [Paper]` [Latent Diffusion Models (2022)](https://arxiv.org/abs/2112.10752). Useful for understanding why latent-space diffusion became so practical.

## Reinforcement Learning and Decision Making

- `[Free] [Intermediate] [Book]` [Reinforcement Learning: An Introduction (2nd Ed., 2018)](http://incompleteideas.net/book/the-book.html). The standard RL text and a solid starting point for reinforcement learning.
- `[Free] [Intermediate] [Book]` [Algorithms for Decision-Making (2022)](https://algorithmsbook.com/). Strong bridge between decision theory, planning, bandits, and RL.
- `[Free] [Intermediate] [Paper]` [AlphaZero (2017)](https://arxiv.org/abs/1712.01815). A concrete example of search plus self-play plus deep learning working together at a high level.
- `[Free] [Intermediate] [Reference]` [Gymnasium](https://gymnasium.farama.org/). Useful once you want a maintained environment API for RL experiments.
- `[Free] [Intermediate] [Paper]` [Direct Preference Optimization (2023)](https://arxiv.org/abs/2305.18290). Good entry point into preference optimization for LLM post-training.

## Symbolic and Hybrid AI

- `[Free] [Intermediate] [Reference]` [Knowledge Representation and Reasoning, Stanford CS227 Notes](https://web.stanford.edu/class/cs227/). Good grounding in logic, ontologies, and formal knowledge representation.
- `[Free] [Beginner] [Course]` [Learn Prolog Now!](https://lpn.swi-prolog.org/). Practical introduction to symbolic programming and rule-based reasoning.
- `[Free] [Advanced] [Survey]` [Towards Cognitive AI Systems: A Neuro-Symbolic Survey (2024)](https://arxiv.org/abs/2402.05123). Useful overview of integration patterns across neural and symbolic systems.
- `[Free] [Advanced] [Paper]` [Logical Neural Networks (2020)](https://arxiv.org/abs/2009.02506). A concrete example of differentiable logic with exact Boolean semantics.

## Interpretability, Safety, and Human Factors

- `[Free] [Intermediate] [Book]` [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/). Practical reference for feature importance, SHAP, LIME, and evaluation caveats.
- `[Free] [Intermediate] [Paper]` [Attention Is Not Explanation (2019)](https://arxiv.org/abs/1902.10186). Useful corrective to common claims about model interpretability.
- `[Free] [Intermediate] [Paper]` [Concrete Problems in AI Safety (2016)](https://arxiv.org/abs/1606.06565). A durable framing of safety problems such as reward hacking and robustness.
- `[Free] [Intermediate] [Book]` [Fairness and Machine Learning (2023)](https://fairmlbook.org/). Strong introduction to the social, legal, and technical dimensions of fairness.
- `[Free] [Intermediate] [Paper]` [Guidelines for Human-AI Interaction (2019)](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/). Still one of the most useful papers for designing systems people can rely on appropriately.
- `[Free] [Intermediate] [Paper]` [Model Cards (2019)](https://arxiv.org/abs/1810.03993). Worth reading if you care about transparency, deployment context, and communicating model limitations.

## What This README Intentionally Leaves Out

- Model leaderboards and "best model" claims.
- Product release roundups and vendor marketing pages.
- Conference schedules, version tables, and community membership counts.
- Very large topic inventories with weak curation.

Those items age quickly and lower the trustworthiness of the guide. If the repository grows, the right move is to add focused topic documents rather than expanding this README indefinitely.

## Contributing

Pull requests are welcome, but the bar is curation, not volume. See [CONTRIBUTING.md](CONTRIBUTING.md) before adding new material.

In practice, that means:

- Prefer one excellent primary source over five overlapping summaries.
- Label each new item with cost, level, and type.
- Add a short reason the resource belongs here.
- Use stable links to the official page, paper, or course when possible.

## Topic Guides

The main README stays intentionally short. Deeper, still-curated lists live in [docs/README.md](docs/README.md):

- [Foundations and Deep Learning](docs/foundations-and-deep-learning.md)
- [Transformers, LLMs, and RAG](docs/transformers-llms-and-rag.md)
- [Interpretability, Safety, and Human Factors](docs/interpretability-safety-and-human-factors.md)

## License

[GPL-3.0](LICENSE)
