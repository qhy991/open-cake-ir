# Isolate Codex author state per Run

Status: accepted for new qualified Codex Runs. This does not qualify clean-start
filesystem read isolation (ADR 0077).

The native CLI reads skills from `CODEX_HOME` even with `--ignore-user-config` and
`skill_search` disabled. A real paired Python qualification on a host with personal
skills emitted an unadmitted skills-budget warning before its file change. An auth-only
home removed that warning in a controlled probe. The author home is therefore a
material input, not an incidental host preference.

`isolated_auth_only_v1` is a Provider configuration commitment. Qualification and
each independent Run start a fresh private home containing only a copied credential.
The CLI may create its own session and system-skill state there; admission refuses
injected user skills, plugins, symlinks and permissive credential custody before
every Turn. The CLI-generated system-skill tree is frozen after the first successful
Turn, compared with the version 2 qualification receipt, and checked before and
after each continuation. A two-arm qualification uses a separate fresh home for
each arm and requires the same system-skill identity. Token refresh may change
`auth.json` bytes; this policy checks private custody rather than claiming a frozen
credential or account identity. The actual invocation receives this home through
`CODEX_HOME`.
The qualification receipt binds the policy name, not credential bytes or a path
that differs per Run. Run-specific homes prevent automatic reuse of previous CLI
sessions as author context. Frozen Provider contracts without this policy retain
their original behavior at their pinned commit.

This policy does **not** stop the author from reading other host paths through a
tool. Python clean-start remains refused until a separate filesystem read jail is
qualified against real target-implementation paths and the actual Provider process.
No live scientific or GPU result follows from the author-home policy alone.
