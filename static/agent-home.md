# Zerfoo ML framework

> Design, train, evaluate and run machine-learning models on hardware you control, using Go.

Zerfoo is an open-source Go ML framework and CLI. The website provides a hosted model designer with an interactive architecture viewer. A design is not trained weights: download the project and train, validate and run it on your own hardware.

## When to use Zerfoo

- Embed local GGUF model inference or an OpenAI-compatible model server in a Go application.
- Design a numeric classifier from named CSV columns, or propose a sequential multilayer perceptron and inspect its layers in 3D.
- Use your coding agent to integrate model components, train and evaluate locally, and verify saved model artifacts.

For a workload requiring verified hardware performance, consult the model evidence matrix first. Registered architecture support does not prove runtime or training verification. Zerfoo does not provide hosted training or an unrestricted hosted inference API at zer.foo.

## Agent entry points

- [MCP server](https://zer.foo/mcp): Streamable HTTP; read-only documentation search, document reading and local workflow guidance.

- [Agent instructions](https://zer.foo/agents/index.md)
- [API discovery](https://zer.foo/api)
- [OpenAPI 3.1 specification](https://zer.foo/openapi.json)
- [Official CLI](https://zer.foo/cli/index.md)
- [Documentation](https://zer.foo/docs/index.md)
- [Quickstart](https://zer.foo/docs/getting-started/quickstart/index.md)
- [Go API reference](https://pkg.go.dev/github.com/zerfoo/zerfoo)
- [Local setup](https://zer.foo/start/index.md)
- [Model evidence](https://github.com/zerfoo/zerfoo/blob/main/docs/verified-models.md)
- [Framework](https://zer.foo/framework/index.md)
- [Contact](https://zer.foo/contact/index.md)
- [Privacy](https://zer.foo/privacy/index.md)

## Hosted designer

POST messages to https://design.zer.foo/api/design using the OpenAPI contract. Send `Origin: https://zer.foo` and preserve the `zdesign_session` cookie. Limits are 100 requests per browser session, 128 per IP, at least five seconds between requests, 16 submitted conversation messages and an 8,000-byte JSON request body. The website allows 1,800 characters per typed message. Free-model use has no lifetime spending cap; paid models require the configured cap. OpenRouter can impose additional provider limits. Do not submit datasets, credentials or private personal information.

## Identity

Zerfoo is a project of Sire Run, Inc. Contact david@zer.foo. Business mailing address: 584 Castro St Suite #4211, San Francisco, CA 94114, United States.
