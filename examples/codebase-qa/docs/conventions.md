# Coding conventions

## Configuration format

This project keeps all configuration in YAML. New config options are added to
`agent.manifest.yaml` and mirrored in the JSON Schema.

## Error handling

Validate only at system boundaries (user input, external APIs). Trust
internal invariants; do not add defensive checks for states that cannot
occur.

## Testing

Every behavioral change ships with a test that would fail without it. Prefer
one pointed assertion over many broad ones.
