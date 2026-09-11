# Product Hunt launch draft

Name: Zerfoo

Tagline: Build AI you own

Description:
Describe a model, export a portable project, then train and run it on hardware you control. Zerfoo is an open-source framework for creating, training, evaluating and running AI, with a growing set of verified workflows.

Maker comment:
Hi, I’m David, the builder of Zerfoo.

I wanted inference to feel like part of a Go application: load a model, call a function, and use the result in the service I’m already building.

Zerfoo is the result. It works as a library, a CLI, or an OpenAI-compatible HTTP server. Default builds are CGo-free, with CUDA loaded at runtime on supported systems.

I’ve published benchmark evidence and a model matrix because architecture support is easy to claim and harder to verify. Start with the model matrix, try the quickstart, and tell me where the experience breaks down for your workload.

I’d especially love feedback from Go developers embedding inference in a real service. What would make this useful enough to adopt?

Website: https://zer.foo/?utm_source=producthunt&utm_medium=launch&utm_campaign=zerfoo
Repository: https://github.com/zerfoo/zerfoo

Gallery assets:
- Homepage hero screenshot: native-to-Go positioning and runtime pipeline.
- Quickstart/code screenshot: the actual library API, not simulated terminal output.
- Performance section screenshot: all four models; include methodology context.
- Social card: static/social-card.png (1200×630).

Before publishing: use the actual deployed URL; choose the date; refresh runtime and model evidence; confirm the current quickstart on the release you link. No launch scheduled or submission made by this redesign.
