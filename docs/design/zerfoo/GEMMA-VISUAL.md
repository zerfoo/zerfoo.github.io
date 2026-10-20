# Full-width Gemma 3 landing visualization

The landing illustration depicts Google Gemma 3 27B, replacing the illustrative
perceptron. It occupies the full hero container below the introduction. Desktop
and mobile SVGs project the same simplified graph horizontally and vertically.

Source: https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/
Google documents text embeddings, SigLIP image encoding, a multimodal projector,
256 soft tokens per crop, 896-square image input, 62 decoder layers for 27B,
5 local layers per global attention layer and 128K context. Final RMSNorm and
LM head are also documented there. The diagram shows one representative attention
group, not all 62 layers; feed-forward blocks are noted but not expanded.
This is an architecture explanation, not a runtime verification result.

Validation: 26 unit tests and ten viewport/theme browser combinations passed.
Browser checks require the diagram width to equal the hero width, asset loading,
no horizontal overflow and preserved landing-to-designer behavior. The local
HTML/Markdown endpoint sweep passed 234 responses across 74 pages.
