"""Enforce the commit convention: ``type(scope): subject``.

Allowed types: feat, fix, chore, docs, refactor, test, perf, ci.
Example scopes: modelling, server, cicd, data, inverse. Merge/revert commits pass.
"""

from __future__ import annotations

import re
import sys

PATTERN = re.compile(
    r"^(feat|fix|chore|docs|refactor|test|perf|ci)(\([a-z0-9\-]+\))?: .+",
)


def main() -> int:
    msg_path = sys.argv[1] if len(sys.argv) > 1 else ".git/COMMIT_EDITMSG"
    with open(msg_path, encoding="utf-8") as fh:
        subject = fh.readline().strip()

    if subject.startswith(("Merge", "Revert")) or PATTERN.match(subject):
        return 0

    sys.stderr.write(
        "\nCommit message must follow: type(scope): subject\n"
        "  e.g. feat(modelling): add quantile forward model\n"
        f"  got: {subject!r}\n\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
