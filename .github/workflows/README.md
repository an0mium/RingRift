# CI/CD Pipeline Documentation

## Overview

RingRift uses a comprehensive CI/CD pipeline to ensure code quality, automated testing, and reliable deployments. The pipeline runs on every push and pull request to the `main` and `develop` branches.

## Pipeline Stages

### 1. Lint (Code Quality Check)

**Purpose**: Ensures code follows consistent style guidelines and best practices.

**What it does**:
- Runs ESLint on all TypeScript files
- Checks for code quality issues, potential bugs, and style violations
- Fails the build if any linting errors are found

**Command**: `npm run lint`

**Fix locally**: `npm run lint:fix`

### 2. Type Check (TypeScript Validation)

**Purpose**: Validates TypeScript types across the entire codebase.

**What it does**:
- Type checks server code (`tsconfig.server.json`)
- Type checks client code (`tsconfig.client.json`)
- Ensures no type errors exist
- Does not emit JavaScript files (--noEmit flag)

**Commands**:
- Server: `npx tsc --noEmit -p tsconfig.server.json`
- Client: `npx tsc --noEmit -p tsconfig.client.json`

### 3. Test (Automated Testing)

**Purpose**: Runs the full test suite with coverage reporting.

**What it does**:
- Executes all unit and integration tests
- Generates code coverage reports
- Uploads coverage to Codecov (optional)
- Archives coverage reports as artifacts
- Fails if coverage thresholds (80%) are not met

**Command**: `npm run test:ci`

**Coverage Artifacts**:
- Available in GitHub Actions for 7 days
- Downloadable HTML report
- LCOV format for external tools

### 4. Build (Production Build)

**Purpose**: Ensures the application can be built successfully.

**What it does**:
- Compiles TypeScript server code
- Builds optimized client bundle with Vite
- Archives build artifacts
- Only runs if lint, typecheck, and test pass

**Commands**:
- `npm run build:server`
- `npm run build:client`

**Build Artifacts**:
- Available in GitHub Actions for 7 days
- Contains compiled server and client code

### 5. Quality Gate (Final Check)

**Purpose**: Final approval step ensuring all quality checks passed.

**What it does**:
- Confirms all previous stages succeeded
- Provides clear success message

## Pre-commit Hooks

Git pre-commit hooks run automatically before every commit to catch issues early.

### What runs on pre-commit:

1. **Lint-Staged**
   - Runs ESLint with auto-fix on staged `.ts` and `.tsx` files
   - Runs Prettier formatting on staged files
   - Only processes files being committed

2. **Type Check**
   - Validates TypeScript types for server and client
   - Ensures no type errors before commit

3. **Related Tests**
   - Runs tests related to changed files
   - Uses `--bail` to stop on first failure
   - Fast feedback loop

### Skipping Pre-commit Hooks

**Not recommended**, but in emergencies:

```bash
git commit --no-verify -m "Emergency fix"
```

## Local Development Workflow

### Before Committing

```bash
# Run all quality checks locally
npm run lint          # Check code style
npm run lint:fix      # Fix code style issues
npm test             # Run all tests
npm run test:coverage # Run tests with coverage
npm run build        # Ensure build works
```

### Recommended Git Flow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit files ...

# 3. Stage changes
git add .

# 4. Commit (pre-commit hooks run automatically)
git commit -m "feat: add new feature"

# 5. Push to remote
git push origin feature/my-feature

# 6. Create pull request
# CI/CD pipeline runs automatically
```

## CI/CD Best Practices

### 1. Write Tests First (TDD)
- Create tests before implementing features
- Aim for 80%+ code coverage
- Test edge cases and error scenarios

### 2. Fix Linting Issues
- Don't commit code with linting errors
- Use `npm run lint:fix` to auto-fix issues
- Configure your editor for real-time linting

### 3. Type Safety
- Avoid `any` types
- Use proper TypeScript types
- Fix type errors, don't bypass them

### 4. Small, Focused Commits
- One feature/fix per commit
- Clear, descriptive commit messages
- Follow conventional commits (feat, fix, docs, etc.)

### 5. Green Build
- Ensure all CI checks pass before merging
- Don't merge failing PR

s
- Fix issues promptly

## Troubleshooting CI/CD Issues

### Lint Failures

**Common causes**:
- Unused variables
- Missing semicolons
- Inconsistent formatting

**Fix**:
```bash
npm run lint:fix
git add .
git commit --amend --no-edit
```

### Type Check Failures

**Common causes**:
- Type mismatches
- Missing type definitions
- Incorrect imports

**Fix**:
- Review TypeScript errors carefully
- Add proper types
- Use type assertions only when necessary

### Test Failures

**Common causes**:
- Race conditions
- Incorrect mocks
- Environment issues

**Fix**:
```bash
npm test -- --verbose
npm test -- tests/unit/failing-test.test.ts
```

### Build Failures

**Common causes**:
- Missing dependencies
- Environment variables
- Module resolution issues

**Fix**:
```bash
npm ci                 # Clean install
rm -rf node_modules    # Nuclear option
npm install
npm run build
```

## Coverage Reports

### Viewing Coverage Locally

```bash
npm run test:coverage
open coverage/lcov-report/index.html
```

### Coverage Thresholds

- **Branches**: 80%
- **Functions**: 80%
- **Lines**: 80%
- **Statements**: 80%

### Improving Coverage

1. Identify uncovered code in the HTML report
2. Write tests for uncovered functions/branches
3. Run `npm run test:coverage:watch` for live feedback

## GitHub Actions Configuration

### Workflow File

Location: `.github/workflows/ci.yml`

### Node.js Version

Currently using Node.js 18 (LTS). Update in workflow file if needed:

```yaml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '18'
```

### Caching

npm dependencies are cached automatically to speed up builds:

```yaml
with:
  cache: 'npm'
```

### Parallelization

Jobs run in parallel when possible:
- `lint` and `typecheck` run simultaneously
- `test` runs independently
- `build` runs after lint, typecheck, and test complete

## Environment Variables

### Required for CI

No environment variables required for basic CI/CD.

### Optional Integrations

- `CODECOV_TOKEN`: For Codecov integration
- Add to GitHub repository secrets if using Codecov

## Maintenance

### Updating Dependencies

```bash
# Check for outdated packages
npm outdated

# Update all dependencies
npm update

# Update specific package
npm install package@latest --save-dev

# After updates, ensure CI passes
npm run test:ci
```

### Updating GitHub Actions

Check for updates to GitHub Actions:
- `actions/checkout`
- `actions/setup-node`
- `codecov/codecov-action`
- `actions/upload-artifact`

## Monitoring

### Build Status

View build status:
- GitHub repository → Actions tab
- Pull request checks
- Commit status badges

### Build History

- Actions tab shows all workflow runs
- Filter by branch, status, or workflow
- Download logs for debugging

## Future Enhancements

Planned CI/CD improvements:

1. **Deployment Automation**
   - Auto-deploy to staging on `develop` branch
   - Auto-deploy to production on `main` branch releases

2. **Performance Testing**
   - Lighthouse CI for frontend performance
   - Load testing for backend APIs

3. **Security Scanning**
   - Dependency vulnerability scanning
   - SAST (Static Application Security Testing)

4. **Release Automation**
   - Semantic versioning
   - Automated changelog generation
   - GitHub releases

---

**Last Updated**: November 13, 2025  
**Maintainer**: RingRift Development Team
