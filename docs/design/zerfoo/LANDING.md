# Landing page and model designer

The homepage introduces Zerfoo, then links to the dedicated `/create/` workspace.
This supersedes the homepage routing described in LIVE-WORKSPACE.md and the first
paragraph of AGENT-READINESS.md. The existing designer main markup is retained;
mobile still places the model before the chat. Shared information-page navigation
links directly to the designer, which includes an Overview link back home.

The design uses the existing Inter typography, dark/light palette, rounded controls
and citron accent. A static neural-network illustration is explicitly labeled as
illustrative. It does not imply a generated or trained model. The framework and
model verification links distinguish ambition from measured capability.

Founder copy is grounded in the local therfoo/therfoo Git history: first commit
24a3979 on January 23, 2019; dense layers, activations and neural-network code
followed that January. This establishes project lineage, not uninterrupted work
or an independent qualification of all current capabilities. Private financial
circumstances and fellowship application details are not public website copy.

The Worker serves actual HTML and negotiated Markdown at `/create/`, replacing
the earlier redirect. Designer analytics initialization is now an external script
so the CSP can keep blocking inline executable scripts. Existing Google fonts and
analytics origins are permitted explicitly. MCP behavior is unchanged.

Validation: 25 unit tests; 10 viewport/theme combinations at 320, 390, 768, 1024
and 1440 pixels; designer chat/download/network-failure browser checks at four
widths; 234 local endpoint responses across 74 pages; 8 real local MCP checks.
Independent source review approved before merge. Production evidence is recorded
separately in the deployment receipt.
