# Model creation launch

The website designs projects; no training, dataset upload or generated-code
execution occurs here. Training and predictions happen on visitor hardware.

## Visual direction

Cloud white #f9fbff, pale blue #f3f7fd, ink blue #173454, action blue #245ac2,
focus amber #d57d1c. Existing self-hosted Inter keeps typography consistent.
Left-aligned conversation and model preview form one workspace. A simple
network diagram is the focal element; no invented metrics or success claims.

## API deployment

Run npm ci, npm test, npx wrangler deploy --dry-run, then deploy.
Set OPENROUTER_API_KEY and a random IP_SALT with Wrangler secrets. Do not commit
credentials. Bind design.zer.foo as the Worker custom domain. Hosted chat stays
disabled until the owner chooses a budget. Set CHAT_ENABLED=true only afterward.
The lifetime ledger reserves one cent before dispatch and never refunds provider
failures. The default maximum is $5 total, not daily. Never reset the durable
object name to bypass its budget. Increase a cap only within owner authorization.
Provider max_price pins input/output upper rates; keep message/output bounds
and per-call reservations consistent when changing models. Workers hosting and
storage charges are separate from this LLM ceiling.

## Current research boundary

The paper library contains unreviewed candidate notes. Do not attach them as
verified research. This version uses the verified numeric-classifier recipe
and says when no reviewed evidence is attached. Research-guided designs remain
an explicit remaining milestone, requiring curated evidence and retrieval tests.

## Migration

Check zer.foo DNS/TLS before changing canonical URLs or GitHub Pages custom
domain. Preserve documentation paths. Redirect zerfoo.feza.ai/* to zer.foo/*
permanently only after the new site responds successfully. Rollback restores
the prior CNAME and DNS configuration. Do not delete unrelated records.
