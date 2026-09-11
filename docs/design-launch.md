# Model creation launch

The website designs projects; no training, dataset upload or generated-code
execution occurs here. Training and predictions happen on visitor hardware.
The project is operated by Sire Run, Inc. at https://sire.run/.

## Visual direction

Cloud white #f9fbff, pale blue #f3f7fd, ink blue #173454, action blue #245ac2,
focus amber #d57d1c. Existing self-hosted Inter keeps typography consistent.
Left-aligned conversation and model preview form one workspace. A simple
network diagram is the focal element; no invented metrics or success claims.

## API deployment

Run npm ci, npm test, npx wrangler deploy --dry-run, then deploy.
Set OPENROUTER_API_KEY and a random IP_SALT with Wrangler secrets. Do not commit
credentials. Bind design.zer.foo as the Worker custom domain. The configured
provider is OpenRouter z-ai/glm-5.3-flash. Hosted chat is enabled with a hard
$20 lifetime design-LLM ceiling (2000 cents), then stops.
The lifetime ledger reserves one cent before dispatch and never refunds provider
failures. The maximum is $20 total, not daily. Never reset the durable object
name to bypass its budget. Increase a cap only within owner authorization.
Provider max_price pins input/output upper rates; keep message/output bounds
and per-call reservations consistent when changing models. Workers hosting and
storage charges are separate from this LLM ceiling.

## Current research boundary

The paper library contains unreviewed candidate notes. Do not attach them as
verified research. This version uses the verified numeric-classifier recipe
and says when no reviewed evidence is attached. Research-guided designs remain
an explicit remaining milestone, requiring curated evidence and retrieval tests.

## Migration

The zone currently has proxied apex records, so the site Worker uses the
`zer.foo/*` zone route and does not delete or replace those records. Preserve
documentation paths. Redirect zerfoo.feza.ai/* to zer.foo/* permanently only
after the legacy zone/Pages custom-domain owner is available; the legacy host
currently still serves GitHub Pages. Rollback removes only the Worker route and
restores the prior CNAME/DNS configuration. Do not delete unrelated records.

## CI deployment

Cloudflare deployment is currently an explicit owner-run step from this
checkout: rebuild Hugo, run `npm test`, then deploy `wrangler.jsonc` and
`wrangler.site.jsonc` separately. The existing GitHub Pages workflow remains
the legacy publisher for `zerfoo.feza.ai` until its owner can install a
permanent redirect to matching `zer.foo` paths. It must not be treated as the
canonical deployment.
