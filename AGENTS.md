# Agent workflow

At the start of every session, read these files in order:

1. `AGENTS.md`
2. `BLOCKERS.md`
3. `docs/model.md`
4. `docs/integration.md`
5. `docs/training.md`
6. `experiments/latest.json`
7. `tasks/today.md`

## Hard rules

- Do not run heavy model training in the IDE. Use the configured external environment.
- Never train, tune, select a champion, or repeatedly evaluate candidates on the held-out test set.
- Confidence is not correctness. Define score interpretation for the actual model type before exposing a score.
- Keep the project lean. Add folders and dependencies only when the project needs them.
- Commit a completed task from `tasks/today.md` only when its description explicitly contains `(commit)`.
- Use a Conventional Commits message when a commit is authorized.
- Do not commit secrets, large datasets, weights, checkpoints, generated artifacts, or external-training credentials.
- Keep training code out of the serving path and preserve preprocessing parity between training and inference.
