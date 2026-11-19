"""Long pastes and text files travel as documents, the way Claude.ai sends them.

The chat UI turns a long paste (or a dropped text file) into an attachment and
sends it ahead of the question, in Anthropic's long-context layout:

    <documents>
    <document index="1">
    <source>Pasted text</source>
    <document_content>
    ...
    </document_content>
    </document>
    </documents>

    question

Why this layout:
- Long material first, question last: models answer noticeably better when
  the question comes after the material it is about.
- Tags keep "what the user pasted" apart from "what the user asked", so a
  log full of "ERROR" is read as a log, not as the user's mood or request.
- The document sits at a fixed place in the prompt, so llama.cpp's prompt
  cache can reuse it on every follow-up instead of reading it again.

index.html builds exactly the same text (wrapDocs), so a read-ahead request
made while the user is still typing shares its prefix with the real one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BLOCK = re.compile(r"\A\s*<documents>\n(.*?)\n</documents>(?:\n\n|\s*\Z)", re.DOTALL)
_DOC = re.compile(
    r'<document index="\d+">\n<source>(.*?)</source>\n<document_content>\n(.*?)\n</document_content>\n</document>',
    re.DOTALL,
)


@dataclass
class Doc:
    name: str
    text: str


def wrap(docs: list[Doc], question: str = "") -> str:
    parts = ["<documents>"]
    for i, d in enumerate(docs, 1):
        parts.append(f'<document index="{i}">\n<source>{d.name}</source>\n'
                     f"<document_content>\n{d.text}\n</document_content>\n</document>")
    parts.append("</documents>")
    return "\n".join(parts) + "\n\n" + (question or "")


def split(text: str) -> tuple[list[Doc], str]:
    """(documents, the user's own words). No documents → ([], text)."""
    t = text or ""
    m = _BLOCK.match(t)
    if not m:
        return [], t
    docs = [Doc(name=n, text=c) for n, c in _DOC.findall(m.group(1))]
    return docs, t[m.end():]


def block_end(text: str) -> int:
    """Index just past the documents block (0 when there is none)."""
    m = _BLOCK.match(text or "")
    return m.end() if m else 0


def question_of(text: str) -> str:
    return split(text)[1]
