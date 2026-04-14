# Security Policy

This is a documentation-only repository containing curated links and annotations. There is no application code, no dependencies, and no deployable software.

## Reporting a Concern

If you discover a security issue (for example, a committed credential or a link pointing to a malicious destination), please email security@lscaturchio.xyz or use [GitHub Security Advisories](https://github.com/gr8monk3ys/ai-resources/security/advisories).

## Measures in Place

- Secret scanning via [Gitleaks](https://github.com/gitleaks/gitleaks) in CI
- Pre-commit hooks for private key detection
- Link validation via [Lychee](https://github.com/lycheeverse/lychee) in CI
