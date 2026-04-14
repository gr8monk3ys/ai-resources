# Prompt Engineering and Agents

Reading list for prompting techniques, tool use, structured generation, and agent architectures. Items in the main [README](../README.md) are not repeated here; start there for the essentials.

Labels:

- Cost: `[Free]`, `[Paid]`
- Level: `[Beginner]`, `[Intermediate]`, `[Advanced]`
- Type: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, `[Reference]`

## Prompting Techniques

- `[Free] [Intermediate] [Paper]` [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022)](https://arxiv.org/abs/2201.11903). Showed that adding reasoning steps to prompts significantly improves performance on multi-step tasks.
- `[Free] [Intermediate] [Paper]` [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (2023)](https://arxiv.org/abs/2305.10601). Extends chain-of-thought with search and backtracking for complex reasoning problems.
- `[Free] [Intermediate] [Paper]` [Self-Consistency Improves Chain of Thought Reasoning in Language Models (2022)](https://arxiv.org/abs/2203.11171). Simple technique of sampling multiple reasoning paths and voting on the answer. Often more effective than single-pass prompting.
- `[Free] [Beginner] [Reference]` [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview). Practical guidance on writing effective prompts with concrete examples and anti-patterns.

## Tool Use and Function Calling

- `[Free] [Intermediate] [Paper]` [Toolformer: Language Models Can Teach Themselves to Use Tools (2023)](https://arxiv.org/abs/2302.04761). Demonstrates how language models can learn to call external tools (search, calculator, APIs) during generation.
- `[Free] [Intermediate] [Paper]` [ReAct: Synergizing Reasoning and Acting in Language Models (2022)](https://arxiv.org/abs/2210.03629). Interleaves reasoning traces with actions, forming the basis for many agent frameworks.

## Structured Generation

- `[Free] [Intermediate] [Reference]` [Outlines](https://github.com/dottxt-ai/outlines). Library for constrained generation using grammars and JSON schemas. Useful when you need structured output from open-weight models.
- `[Free] [Intermediate] [Reference]` [Instructor](https://github.com/jxnl/instructor). Structured extraction from LLM outputs using Pydantic models. Works with both API-based and local models.

## Agent Architectures

- `[Free] [Intermediate] [Reference]` [LLM Powered Autonomous Agents (2023)](https://lilianweng.github.io/posts/2023-06-23-agent/). Practical systems-oriented guide to memory, planning, and tool use in agents.
- `[Free] [Advanced] [Survey]` [A Survey on LLM-based Autonomous Agents (2023)](https://arxiv.org/abs/2308.11432). Useful map of agent architectures, memory, planning, and tool use.
- `[Free] [Intermediate] [Paper]` [Voyager: An Open-Ended Embodied Agent with Large Language Models (2023)](https://arxiv.org/abs/2305.16291). Concrete example of an LLM agent that writes and reuses code in an open-ended environment.

## Study Order

1. Read `Chain-of-Thought` and `Self-Consistency` to understand basic prompting beyond zero-shot.
2. Read `ReAct` and `Toolformer` to understand how models interact with external tools.
3. Use the Anthropic prompt guide as a practical reference while building.
4. Read the agent survey and `LLM Powered Autonomous Agents` before designing agent systems.
5. Pick up `Outlines` or `Instructor` when you need reliable structured output from models.
