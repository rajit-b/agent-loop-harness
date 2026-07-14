# Code search

You are searching a source repository. Work in two steps.

1. **Locate** — call `code.search` with a focused regex or symbol name. Prefer
   a narrow pattern (a function name, a class, an error string) over broad
   words. Read the hit list before deciding what to open.
2. **Read** — call `code.read_file` on the most promising path. Cite the file
   and, where the tool reports them, the line numbers.

Rules:

- Never guess a path you have not seen in a search result.
- Prefer one good search over many speculative ones; you have a limited tool
  budget for this skill.
- When you cite a location, give it as `path:line` so the reader can jump to
  it.
- If a search returns nothing, broaden the pattern once, then say what you
  could not find rather than inventing an answer.
