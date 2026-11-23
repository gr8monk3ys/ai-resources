# AI Learning Resources

A curated, deliberately short list of books, papers, courses, and reference material for learning AI and machine learning, kept as Markdown.

## Language

**Resource**:
One linked item in a list: a book, course, paper, survey, or reference, with a label triple and a one-sentence reason it belongs.
_Avoid_: link, entry, bookmark

**Label**:
The bracketed metadata prefixing every Resource, in the order Cost, Level, Type (for example `[Free] [Intermediate] [Paper]`).
_Avoid_: tag, badge

**Cost**:
The first Label: `[Free]` or `[Paid]`.

**Level**:
The second Label: `[Beginner]`, `[Intermediate]`, or `[Advanced]`.

**Type**:
The third Label: `[Book]`, `[Course]`, `[Paper]`, `[Survey]`, or `[Reference]`.

**Curation Principles**:
The README rules that decide what gets in: primary sources, free where possible, short enough to skim, no hype or fast-aging product detail.
_Avoid_: guidelines, criteria

**Start Here**:
The README table mapping a learner goal to a suggested reading order of Resources.
_Avoid_: roadmap, learning path

**Prerequisites**:
The README section of Beginner Resources for programming and math background, ahead of the AI material.

**README**:
The short, high-trust main list at the repo root. It stays intentionally brief and links out to Topic Guides for depth.
_Avoid_: index

**Topic Guide**:
A deeper, still-curated list in `docs/` for one area (Foundations and Deep Learning; Transformers, LLMs, and RAG; Interpretability, Safety, and Human Factors). It uses the same Label format as the README.
_Avoid_: appendix, extended list

## Relationships

- A **Resource** carries exactly one **Cost**, one **Level**, and one **Type**.
- The **README** and every **Topic Guide** list Resources; a Topic Guide adds depth, not volume.
- **Curation Principles** govern what any list may contain.
