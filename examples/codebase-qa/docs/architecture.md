# Architecture

## The config loader

The configuration loader lives in `src/app/config/loader.py`. It reads the
manifest, then applies resolution precedence: built-in defaults first, then
the manifest file, then environment variables, and finally CLI flags. The
last writer wins for scalars; maps deep-merge; lists replace wholesale.

## Validation

Unknown keys fail closed — a typo in the manifest is a hard error, not a
silent no-op. The loader validates eagerly at startup so misconfiguration
never reaches the running agent.

## The agent loop

The loop is an explicit state machine: perceive, retrieve, plan, act,
observe, reflect, terminate. Budgets (steps, tokens, wall clock, cost) are
checked at transition boundaries, and exceeding any of them forces a
graceful wrap-up.
