# Contributing Guide

## Pre-commit Checks

Before committing, ensure your code passes all checks:

### Automatic (via Git Hook)

A pre-commit hook is installed that automatically:
- Formats code with `cargo fmt`
- Checks formatting
- Runs `cargo clippy`

The hook will prevent commits if any check fails.

### Manual Check

You can also run checks manually:

```bash
# Run the pre-commit check script
./scripts/pre-commit-check.sh

# Or run individual checks
cargo fmt --all
cargo clippy --all-features -- -D warnings
cargo test --lib
```

## Code Quality Standards

- ✅ All code must be formatted with `cargo fmt`
- ✅ All code must pass `cargo clippy --all-features -- -D warnings`
- ✅ All tests must pass: `cargo test --lib`
- ✅ Documentation tests must pass: `cargo test --doc`

## CI Checks

The CI pipeline will:
- Check code formatting
- Run clippy lints
- Run unit tests on Linux and macOS
- Run security audit
- Build documentation

Make sure all checks pass before pushing!

