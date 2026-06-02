# Contributing

This is a small learning project. Contributions should keep the code easy to read and easy to run.

Good contributions:

- clearer explanations
- small tests
- safer config defaults
- simpler training utilities
- focused bug fixes
- visual examples that teach the training process

Please avoid:

- large unrelated rewrites
- adding generated datasets or checkpoints
- committing private cloud project values
- adding provider-specific secrets or local machine paths

## Local Checks

```bash
python -m pytest
python -m ruff check .
```

## Style

- Prefer simple code over clever abstractions.
- Keep scripts composable from the command line.
- Keep configs explicit.
- Document any tensor shape assumptions near the code that depends on them.

