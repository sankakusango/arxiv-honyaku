"""TeXコメントのうち, 翻訳不要な行頭コメントを除去するモジュール."""

from __future__ import annotations

import re

_BEGIN_ENVIRONMENT_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_ENVIRONMENT_RE = re.compile(r"\\end\{([^}]+)\}")
_VERBATIM_LIKE_ENVIRONMENTS = frozenset(
    {
        "lstlisting",
        "minted",
        "verbatim",
        "verbatim*",
    }
)


def strip_comment_only_lines(text: str) -> str:
    """翻訳に不要な行頭コメント行を除去する.

    verbatim 系環境の内部は, ``%`` がコード本文として意味を持ちうるため,
    そのまま保持する.

    Args:
        text: 元のTeX本文.

    Returns:
        str: 行頭コメント行を除去したTeX本文.
    """
    lines = text.splitlines(keepends=True)
    kept_lines: list[str] = []
    verbatim_environment_stack: list[str] = []

    for line in lines:
        for match in _BEGIN_ENVIRONMENT_RE.finditer(line):
            environment_name = match.group(1).strip()
            if environment_name in _VERBATIM_LIKE_ENVIRONMENTS:
                verbatim_environment_stack.append(environment_name)

        if not verbatim_environment_stack and line.lstrip().startswith("%"):
            for match in _END_ENVIRONMENT_RE.finditer(line):
                environment_name = match.group(1).strip()
                if (
                    verbatim_environment_stack
                    and verbatim_environment_stack[-1] == environment_name
                ):
                    verbatim_environment_stack.pop()
            continue

        kept_lines.append(line)

        for match in _END_ENVIRONMENT_RE.finditer(line):
            environment_name = match.group(1).strip()
            if (
                verbatim_environment_stack
                and verbatim_environment_stack[-1] == environment_name
            ):
                verbatim_environment_stack.pop()

    return "".join(kept_lines)
