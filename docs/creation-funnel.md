# Creation funnel

GA4 measurement ID: G-EDB1KGM9VZ.

| Event | Meaning | Parameters |
|---|---|---|
| chat_started | First nonempty submission per page conversation, including requests that fail | none |
| design_generated | Service returned a validated project and the UI rendered its preview | design_task |
| project_downloaded | Browser download was initiated | design_task |
| design_failed | Chat request or response handling failed | failure_reason |

`design_task` is numeric_classification, component_design, design_brief or unknown.
`failure_reason` is network, rate_limited, unavailable, provider_unavailable,
provider_timeout, invalid_request, invalid_response, invalid_design or unknown.
The allowlist drops all other parameters. It never sends chat text, model objectives,
feature/target names, provider errors, ZIP contents or custom user identifiers.
Normal GA page measurement still applies. Blocked analytics never blocks the product.

In GA4, use an ordered funnel exploration for chat_started → design_generated →
project_downloaded. Compare users reaching each step, not raw event totals:
refinements can generate several designs and downloads per conversation.
Use design_failed counts to locate breakage; register design_task and failure_reason
as event-scoped custom dimensions if parameter breakdowns are needed in reports.
This change emits events; it does not create GA report configuration.
A download event proves initiation only, not file saving, model training or success.
No measurement is available for visitors who block analytics.
