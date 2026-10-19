"""Собрать самостоятельные UTF-8 HTML-версии учебника и ответов на замечания."""

import html
import re
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("answers_to_feedback.md", "project_manual.md")
STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #172033; }
body { margin: 0; background: #edf2f7; line-height: 1.7; }
main { max-width: 1000px; margin: 24px auto; padding: 36px; background: white; }
nav, details { padding: 16px; background: #eff6ff; border-radius: 8px; }
nav a { margin-right: 16px; }
.contents > li { margin: 12px 0; font-weight: 700; }
.contents ul { margin-top: 4px; padding-left: 22px; font-weight: 400; }
h1, h2, h3, h4 { line-height: 1.3; scroll-margin-top: 16px; }
h2 { margin-top: 2.5em; border-bottom: 2px solid #dbeafe; padding-bottom: 8px; }
a { color: #174ea6; overflow-wrap: anywhere; }
code { font-family: ui-monospace, monospace; background: #f1f5f9; padding: 2px 4px; }
pre { background: #f1f5f9; padding: 16px; overflow-x: auto; line-height: 1.5; }
pre code { padding: 0; }
table { display: block; overflow-x: auto; border-collapse: collapse; margin: 20px 0; }
th, td { border: 1px solid #ccd6e0; padding: 9px 12px; text-align: left; }
th { background: #eff6ff; }
blockquote { border-left: 4px solid #3b82f6; margin-left: 0; padding: 8px 20px; background: #f8fafc; }
summary { cursor: pointer; font-weight: 700; }
.help { color: #475569; font-size: 0.95em; }
@media (max-width: 700px) { main { margin: 0; padding: 16px; } }
@media print {
  body { background: white; font-size: 10pt; }
  main { margin: 0; padding: 0; max-width: none; }
  nav, details, .help { display: none; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; }
  table { display: table; font-size: 9pt; }
  h2, h3 { break-after: avoid; }
}
"""


class DocumentRenderer(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__(escape=True)
        self.headings = []
        self.occurrences = {}

    def heading(self, text, level, **attrs):
        label = html.unescape(re.sub(r"<[^>]+>", "", text))
        slug = re.sub(r"[^\w\s-]", "", label.lower())
        slug = re.sub(r"\s+", "-", slug.strip()) or "section"
        count = self.occurrences.get(slug, 0)
        self.occurrences[slug] = count + 1
        anchor = f"{slug}-{count}" if count else slug
        self.headings.append((level, label, anchor))
        return f'<h{level} id="{html.escape(anchor, quote=True)}">{text}</h{level}>\n'

    def link(self, text, url, title=None):
        path, separator, fragment = url.partition("#")
        if path in DOCUMENTS:
            url = str(Path(path).with_suffix(".html")) + (separator + fragment if separator else "")
        return super().link(text, url, title)


def render_document(text):
    renderer = DocumentRenderer()
    markdown = mistune.create_markdown(renderer=renderer, plugins=["table"])
    body = markdown(text)
    if not renderer.headings or renderer.headings[0][0] != 1:
        raise ValueError("Документ должен начинаться с заголовка первого уровня")
    title = html.escape(renderer.headings[0][1])
    chapters = []
    for level, label, anchor in renderer.headings:
        link = f'<a href="#{html.escape(anchor, quote=True)}">{html.escape(label)}</a>'
        if level == 2:
            chapters.append((link, []))
        elif level == 3 and chapters:
            chapters[-1][1].append(f'<li>{link}</li>')
    contents = "\n".join(
        f'<li>{link}' + (f'<ul>{"".join(children)}</ul>' if children else '') + '</li>'
        for link, children in chapters
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{STYLE}</style>
</head>
<body><main>
<nav aria-label="Документы"><a href="project_manual.html">Полный учебник</a>
<a href="answers_to_feedback.html">Ответы на замечания</a>
<a href="../README.md">README проекта</a></nav>
<p class="help">Откройте этот файл в браузере. Поиск: Ctrl+F / ⌘F.
Печать или сохранение в PDF: Ctrl+P / ⌘P. Интернет для чтения не нужен.
Исходники Markdown удобнее читать на GitHub или в предпросмотре PyCharm.</p>
<details open><summary>Оглавление</summary><ol class="contents">{contents}</ol></details>
{body}
</main></body>
</html>
"""


def main():
    for name in DOCUMENTS:
        source = ROOT / "docs" / name
        output = source.with_suffix(".html")
        output.write_text(render_document(source.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"Создан {output.relative_to(ROOT)} (UTF-8)")


if __name__ == "__main__":
    main()