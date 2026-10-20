# Zerfoo agent readiness

The homepage remains a chat/model workspace. Agent representations are negotiated
at the edge; there is no browser redesign or new public training service.

## Coverage

- Markdown content negotiation on the homepage and content pages: actual Markdown,
  `Content-Type: text/markdown; charset=utf-8`, `Vary: Accept`, alternate links and
  a separate HTML representation. Direct `index.md` resources are generated too.
- Missing pages retain HTTP 404, with a Markdown explanation and discovery links
  when requested. API discovery and unknown operations use JSON.
- OpenAPI 3.1.1 describes the real hosted designer, runtime downloads and public
  resource discovery, including operation IDs, typed schemas, Origin/cookie
  requirements, limits, response contracts and resolution hints.
- The designer's structured errors preserve the browser-compatible `error` string
  and add stable `code`, `message`, `hint` and documentation fields.
- `llms.txt` and agent instructions identify best-fit jobs, invocation and recovery,
  evidence limits, CLI installation, API contract and developer references.
- Homepage metadata includes the product identity, canonical URL, language,
  Open Graph image/type and SoftwareApplication/Organization JSON-LD. Maintainer
  email and business mailing address were supplied by the owner.
- Contact and Privacy pages describe real channels and implemented data flow.
- The official Homebrew formula builds the public CLI's pinned v1.60.0 source.
  The separate proprietary creation runtime is not republished by that formula.

## Validation

`npm test` covers negotiation quality, 404 bodies, JSON errors, HEAD, redirects,
Markdown conversion, OpenAPI validation, identity metadata and trust pages.
`node scripts/verify-agent-endpoints.mjs [origin]` verifies all generated public
page representations, machine-readable files, schema parsing and error behavior.
Run it after `npm run build` and again after deployment. It writes its receipt to
`/tmp/zerfoo-agent-verification.json`; a local pass is not a production receipt.

The native CLI source build was checked with CGo disabled. Homebrew CI installs
and tests the actual formula independently. Consult the public tap's run result
for package-installation evidence.

## Limits of this work

The audit score must be rerun by the auditor; no new score is inferred from source
or endpoint tests. Search ranking is external and varies by query and time.
Search Console ownership, index submission, outreach and press mentions require
the relevant accounts and owner-directed publication. Do not invent trust claims,
reviews, addresses or performance evidence to improve an audit score.
