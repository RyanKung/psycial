#!/bin/bash
# Pre-commit check script - run this before committing
# Usage: ./scripts/pre-commit-check.sh

set -e

echo "🔍 Running pre-commit checks..."

# Format code
echo "📝 Formatting code..."
cargo fmt --all

# Check formatting
echo "✅ Checking formatting..."
if ! cargo fmt --all -- --check; then
    echo "❌ Formatting check failed!"
    exit 1
fi

# Run clippy
echo "🔧 Running clippy..."
if ! cargo clippy --all-features -- -D warnings; then
    echo "❌ Clippy check failed!"
    exit 1
fi

# Run tests
echo "🧪 Running tests..."
if ! cargo test --lib; then
    echo "❌ Tests failed!"
    exit 1
fi

# Check documentation warnings
echo "📚 Checking documentation..."
DOC_OUTPUT=$(cargo doc --no-deps 2>&1)
if echo "$DOC_OUTPUT" | grep -qE "warning|error"; then
    echo "❌ Documentation has warnings or errors!"
    echo "$DOC_OUTPUT" | grep -E "warning|error" | head -20
    exit 1
fi

echo "✅ All pre-commit checks passed!"

