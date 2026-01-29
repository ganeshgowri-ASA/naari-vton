# Lessons Learned from Jewelry VTON Training Pipeline Debugging

## Lesson 1: Always Verify Error Root Cause Before Taking Action

- **What Happened**: Workflow runs #1-16 failed. After seeing "insufficient credit" error in run #16, we immediately assumed credits were the issue and purchased $10 Replicate credits
- **Actual Problem**: The Docker build was failing with `pyenv: not found` error (exit code 127) BEFORE reaching the training step where credits would matter
- **Impact**: Wasted time and money troubleshooting the wrong issue
- **Prevention**:
  - Always examine build logs thoroughly from the beginning
  - Look for exit codes and error messages at each step
  - Don't jump to conclusions based on one error message
  - Verify the execution flow - if build fails, subsequent errors are irrelevant

## Lesson 2: Distinguish Between Build-Time vs Runtime Errors

- **What Happened**: Mixed up Docker build configuration errors with Replicate API runtime errors
- **Key Difference**:
  - Build errors (exit 127, command not found) = Docker container setup failing
  - Runtime errors (insufficient credit, API errors) = Application logic failing after successful build
- **Prevention**: Check which phase the error occurs in: build vs deploy vs runtime

## Lesson 3: Use Systematic Debugging for CI/CD Pipelines

- **Correct Order**:
  1. Check workflow YAML syntax
  2. Verify Docker build completes successfully
  3. Check environment variables and secrets
  4. Verify API credentials and quotas
  5. Test application logic
