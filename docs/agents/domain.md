# Domain Docs

## Before exploring, read these

- `CONTEXT.md` at the repository root.
- `CONTEXT-MAP.md` if it exists.
- Relevant ADRs under `docs/adr/`.

If these files do not exist, proceed silently. Domain-modeling skills create them only when the project has terminology or decisions worth recording.

## Layout

This is a single-context repository:

```text
/
├── CONTEXT.md
├── docs/adr/
└── src/
```

## Vocabulary

Use domain terms as defined in `CONTEXT.md`. Do not drift to rejected synonyms. If a needed concept is absent, reconsider the term or flag it for `/domain-modeling`.

## ADR conflicts

Surface conflicts with existing ADRs explicitly rather than silently overriding them.
