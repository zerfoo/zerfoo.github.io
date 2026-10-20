# Three.js

Vendored ES modules from https://github.com/mrdoob/three.js at commit
5789a76104d4a9b68fd037a5f9caaeb3eafe82b5 (package 0.186.0, development build).
MIT license: see LICENSE. OrbitControls' bare `three` import is rewritten to
`./three.module.js`; one indentation whitespace issue in three.core.js is normalized. No runtime logic is changed.

Used only for an optional, lazily loaded architecture illustration. The accessible
HTML model details remain available if WebGL is unavailable. No model execution.
