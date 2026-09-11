# Web model creation evidence

| Check | Result |
| --- | --- |
| Core revision in download | `bfbb707111127cedb017d741f8b9337da1ed8632` |
| Exported lifecycle at handoff | Iris training 20 epochs / 120 steps; validation 93.3%; fresh prediction probability `.997919`; ZIP integrity passed |
| Hosted budget | Durable Object lifetime ledger, 2000-cent cap, one-cent reservation |
| Research | 897 candidate notes remain unreviewed; reviewed allowlist is empty |
| Kazi handoff | Converged with installed Kazi and free OpenCode model; HTTP probe header bug reported as issue #1855 |
| Live browser conversation | Passed against `https://zer.foo/create/`; deployed API returned a ready numeric-classification project and browser download `zerfoo-project.zip` passed ZIP integrity |
| Browser and domain | Browser pass; apex custom-domain attachment blocked by existing proxied A records; switched to `zer.foo/*` zone route |
| Live deployments | API version `1190abaf-e9f4-4734-b408-eab5af4e75c3`; site version `c84d1b8f-988d-44c3-bff5-2fec3bd87595`; required live paths return 200 |
| Legacy hostname | `zerfoo.feza.ai` still serves GitHub Pages (200); permanent redirect requires legacy zone/Pages ownership and remains open |

This file contains no credentials, private paths or dataset rows.
