# Zerfoo website

Hugo marketing site and documentation for https://zerfoo.feza.ai/.

## Preview

```sh
git submodule update --init --recursive
hugo server
```

Use Hugo extended 0.159.0 and Dart Sass, matching the existing Pages workflow. Build with `hugo --minify`. Pushing main triggers the existing GitHub Pages deployment.

Homepage: `content/_index.html`. Plain CSS: `static/css/site.css`. Interaction code: `static/js/site.js`. Docs tokens: `assets/_custom.scss`. Brand decisions and evidence: `docs/design/zerfoo/DESIGN.md`. Product Hunt copy: `docs/product-hunt-draft.md`.

Browser checks use `scripts/check-site.cjs` with an installed Playwright module. Set `PLAYWRIGHT_MODULE` to its path if necessary. Start the local preview on port 4879 first. Screenshots and check results go to `renders/`.

The landing page intentionally uses the supplied dark art direction under either system preference. Documentation supports light and dark themes. No frontend framework or production JavaScript dependencies are required.
