# Digest

Turn a long piece of text into a structured, skimmable digest. The heavy
lifting — reading and condensing — is done by the `summarizer.summarize`
tool, which is backed by a Claude Sonnet call. Your job is to orchestrate it
and shape the output.

## Steps

1. **Condense.** Call `summarizer.summarize` on the source text with
   `style: "bullet points"`. If the user named a topic they care about, pass
   it as `focus` so the summary emphasizes it.
2. **Structure.** From the returned summary, produce three sections:
   - **TL;DR** — a single sentence capturing the whole thing.
   - **Key points** — 3–6 bullets, most important first.
   - **Action items** — concrete next steps, each as `- [ ] …`. Omit this
     section entirely if the text implies no actions.
3. **Ground every claim.** Include only facts that appear in the summary or
   the source. Do not add outside knowledge. If the text is too short or
   empty to summarize, say so plainly instead of padding.

## Notes

- One `summarize` call is usually enough. Only call it again if the user asks
  for a different angle (e.g. a re-summary focused on risks). You have a small
  tool budget for this skill.
- Keep the final digest tight. Brevity is the point.
