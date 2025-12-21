from __future__ import annotations

from typing import Iterable


_GLOBAL_FLAGS = {"-v", "--verbose"}
_GLOBAL_KV = {"--config", "--output-dir", "--seed"}


def split_global_args(argv: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split run_tournament global args from subcommand args."""
    global_args: list[str] = []
    subcommand_args: list[str] = []
    args = list(argv)
    idx = 0

    while idx < len(args):
        arg = args[idx]
        if arg in _GLOBAL_FLAGS:
            global_args.append(arg)
            idx += 1
            continue
        if arg in _GLOBAL_KV:
            if idx + 1 < len(args):
                global_args.extend([arg, args[idx + 1]])
                idx += 2
                continue
        subcommand_args.append(arg)
        idx += 1

    return global_args, subcommand_args
