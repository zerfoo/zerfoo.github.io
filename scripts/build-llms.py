"""Regenerate the expanded documentation corpus after content edits."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
parts = [(root / 'static/llms.txt').read_text(), '\n# Documentation corpus\n\nGuides and historical engineering articles follow. For current support and performance, use the evidence links above.\n']
for path in sorted((root / 'content/docs').rglob('*.md')):
    text = path.read_text()
    if text.startswith('---\n'):
        text = text.split('---', 2)[-1].strip()
    relative = path.relative_to(root / 'content').with_suffix('')
    url = str(relative).removesuffix('/_index')
    parts.append(f'\n\n---\nSource: https://zer.foo/{url}/\n\n{text}')
(root / 'static/llms-full.txt').write_text('\n'.join(parts) + '\n')
