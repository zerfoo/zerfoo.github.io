# Live model workspace

The homepage now shows chat on the left and the current designed model on the
right across the viewport. Below 761px the model precedes the chat. Marketing,
documentation, coding-agent handoff and existing navigation remain available.

The client POSTs the existing credentialed conversation to the design API. It
renders the actual returned version-1 project and downloads the server-generated
ZIP. It never synthesizes a sample project or infers a graph from prose.

Numeric classification uses the existing `dense-relu-16-v1` contract: exact input
features, Dense(16), ReLU and an output whose class count is discovered locally.
The output node is explicitly an unknown-dimension placeholder. Component designs
show selected component IDs and architecture prose; links/shapes remain unspecified.
Design briefs show an empty architecture state. No trained weights or live
predictions are represented. Animation is illustrative only.

Conversation-only turns and failures preserve the prior model, labeled as the
last design. Failed sends restore the prompt without entering failed turns into
request history. Requests are bounded and not automatically retried. New design
clears browser conversation/project state but does not clear cookies or reset API
budgets. Only allowlisted event categories are sent to analytics.

Three.js is vendored with its MIT license and revision in `static/vendor/three`.
It loads only when a numeric design exists. HTML layer details are also available;
WebGL failures select the HTML view. Reduced motion uses layer stepping. Existing
site theme changes update the scene.

## Acceptance before deployment

- Seven Node tests: adapter shapes, unknown output count, input rejection,
  component/brief distinctions, success/clarification/failure/reset history,
  server ZIP handling, fixed-category analytics and private-source boundary.
- Hugo build and Wrangler dry-run succeed. Hugo/Sass issue existing deprecation
  warnings. No backend/provider, budget, DNS or infrastructure changes.
- Independent source review found no blocking issues; reset handoff text fixed.
- Browser checks distinguish controlled local API fixtures from live provider
  acceptance. Production checks are recorded after deployment below.

The original `/create/` route redirects to the homepage through the Worker; its
static HTML also redirects for direct static hosting.
