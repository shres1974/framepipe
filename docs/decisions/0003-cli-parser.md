# 0003 In-repo CLI parser

Status: accepted, 2026-11-03

The spec allows "a pinned small library or a small documented parser". A
~100-line parser (`src/app/cli_args.cpp`) avoids another dependency.

Grammar: `framepipe <command> [--flag] [--option VALUE | --option=VALUE]...`

- Unknown options, repeated options, missing values, and values given to
  flags are errors (exit 2).
- Every command supports `--help` (stdout, exit 0). Help text is generated
  from the same option table used for parsing.
- Integers are parsed fully (`--width 12x` is rejected).
- Diagnostics go to stderr. stdout carries only requested JSON
  (`doctor --json`, `--report -`).
