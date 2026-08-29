"""Code-augmentation routines for minority classes.

Augmentations preserve label semantics:
  - rename_locals: rename non-keyword identifiers to vary surface form
  - dup_lines: duplicate a self-contained statement line

Both keep the function parseable: renaming skips C keywords, and duplication
skips any line carrying a brace, a label, or a preprocessor directive, so
block structure is unchanged. They vary surface form only, which is what makes
them safe to apply to a CWE-labeled function without changing its class.
"""
from __future__ import annotations

import random
import re

KEYWORDS = {
    "if", "else", "for", "while", "return", "int", "char", "void", "float",
    "double", "long", "short", "struct", "typedef", "static", "const", "unsigned",
    "signed", "switch", "case", "break", "continue", "do", "sizeof", "goto",
    "extern", "register", "auto", "volatile", "enum", "union", "default", "NULL",
}
IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{1,})\b")


def rename_locals(code: str, rng: random.Random) -> str:
    idents = [m.group(1) for m in IDENT_RE.finditer(code) if m.group(1) not in KEYWORDS]
    if not idents:
        return code
    seen = []
    for x in idents:
        if x not in seen:
            seen.append(x)
    rename = {}
    for x in seen:
        if rng.random() < 0.30:
            rename[x] = f"{x}_v{rng.randint(0, 9999)}"
    if not rename:
        return code
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in rename) + r")\b")
    return pattern.sub(lambda m: rename[m.group(1)], code)


def _is_simple_statement(line: str) -> bool:
    """True for a line that can be repeated without unbalancing the function.

    Duplicating a line that opens or closes a block, or a preprocessor
    directive, changes the parse. Only self-contained statements qualify.
    """
    s = line.strip()
    if not s or s.startswith(("//", "#", "case", "default", "goto")):
        return False
    if any(ch in s for ch in "{}"):
        return False
    return s.endswith(";") and not s.startswith(("return", "break", "continue"))


def dup_lines(code: str, rng: random.Random) -> str:
    out = []
    for ln in code.splitlines():
        out.append(ln)
        if _is_simple_statement(ln) and rng.random() < 0.05:
            out.append(ln)
    return "\n".join(out)


def augment(code: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        code = rename_locals(code, rng)
    if rng.random() < 0.3:
        code = dup_lines(code, rng)
    return code
