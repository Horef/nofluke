# AGENTS.local.md template

Copy to `AGENTS.local.md` (git-ignored) and fill it in for this machine; agents create and update it. One `- key: value` line per fact. Tracked files refer to these keys, never to their values. No secrets: agents read this file into their context.

- machine: <host name (OS, arch, CPUs, RAM)> - from `python tools/docmap.py local`
- python: <absolute path of the project's Python 3.12 interpreter, e.g. <repo>/.venv/bin/python> - what `python` in AGENTS.md means
- memory_budget: <peak RAM that local runs (tests, benchmarks) may use, e.g. "6 GB, never above 10 GB">
- compute: <where heavy benchmark sweeps may run, or "none">
- notes: <quirks agents must respect here, e.g. the SSH key asks for a passphrase, so push unpiped>
