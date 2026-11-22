---
status: accepted
date: 2026-11-22
---

# A short curated README plus Topic Guides, in plain Markdown

The repo is a reading list, not a directory or news feed. It is plain Markdown with no build step, data file, or site generator: a short root `README.md` for skimming, and focused guides in `docs/` for depth. Every Resource carries a Cost/Level/Type Label and a one-line reason.

Why: a short list stays trustworthy as hype ages, and plain Markdown renders on GitHub with nothing to maintain or break. The README itself says that when the list grows, the right move is another focused Topic Guide rather than a longer README.

## Consequences

- No generated pages, search, or filtering by Label; readers scan the files.
- Label consistency is enforced by contributors and review, not by tooling. Adding a lint or script would be a new decision.
