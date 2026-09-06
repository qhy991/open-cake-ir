# Security

Report a suspected security vulnerability privately through the repository's
[Security page](https://github.com/qhy991/open-cake-ir/security), using **Report a vulnerability**.
Include the affected source revision, input, environment, and a minimal reproduction.
Do not place credentials, private datasets, or exploit details in a public issue.

This is a research system. Running generated kernels, evaluators, or external compiler
tools executes code in the configured environment. IR verification checks its modeled
program contracts; it is not a general security sandbox for arbitrary GPU or host code.
Use an appropriate isolated environment for untrusted contributions and experiments.

The currently maintained source is `main`. Historical releases and research branches are
retained for replay and development; they are not a promise of ongoing security support.
