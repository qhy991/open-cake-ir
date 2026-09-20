# Running open-cake-ir tasks on bw1100 (gfx938)

*Operator notes for one specific host, not a contract. The scripts it names live under
`/tmp` on bw1100 and are not in this repository; this file records what they are and how to
drive them so the knowledge does not live only in a session transcript. Nothing here is
evidence — what the campaigns produced is under `findings/`, and the reading of it is
`docs/dcu-gfx938-results.md`. The design and the gate sequence are in
`docs/dcu-gfx938-design.md`.*

*The provider token is supplied through the environment by whoever runs a campaign. It is
not in this repository, not in any workspace, and not in the container image.*

Three scripts, all already on bw1100 under `/tmp`. Start here:

```bash
ssh bw1100 'setsid bash /tmp/sweep_dcu.sh > /tmp/sweep.log 2>&1 < /dev/null & disown'
```

Detached on purpose: bw1100's link drops often enough that a foreground run dies with the
ssh session. Poll the log; don't hold the connection.

## The three scripts

| script | what it does |
|---|---|
| `/tmp/run_dcu_task.sh` | one task, end to end, in the DTK container |
| `/tmp/sweep_dcu.sh` | a set of tasks in sequence, with a ledger |
| `/tmp/read_outcome.sh` | one workspace's adherence and disposition |

### One task

```bash
bash /tmp/run_dcu_task.sh <task> [turns] [budget] [model]
bash /tmp/run_dcu_task.sh rmsnorm 2 200000 deepseek-v4-flash   # the defaults
```

### A sweep

```bash
bash /tmp/sweep_dcu.sh                              # the default twelve
bash /tmp/sweep_dcu.sh rmsnorm silu gemm            # only these
TURNS=4 BUDGET=400000 bash /tmp/sweep_dcu.sh rmsnorm
MODEL=deepseek-v4-pro bash /tmp/sweep_dcu.sh gemm
```

Writes `/home/testuser01/oci-dcu-runs/sweep-<stamp>.tsv` and a per-task log directory
beside it. The ledger's `adherence` column comes from each run's own sealed terminal
event, not from the exit code -- a task can exit non-zero having produced a good run, and
a tolerated fault would otherwise never show up.

The sweep re-execs from a snapshot of itself, so editing it while it runs is safe. bash
reads a script incrementally from the current offset; an earlier sweep died on its last
task because the file changed underneath it.

### Reading a run afterwards

```bash
bash /tmp/read_outcome.sh rmsnorm-20260917-174616     # -> "adhered evaluated"
```

Workspaces are written by root inside the container, so the host account cannot read them
directly; this script reads them back through the same image.

## The 29 tasks

```
base         rmsnorm layernorm residual_rmsnorm softmax
activation   gelu_tanh gelu_tanh_backward prelu selu silu softplus_gradient softsign swiglu
rowwise      absmax_rescale cosine_similarity layernorm_backward_input
             rmsnorm_input_gradient softmax_backward
reduction    bias_gradient_reduction channel_absmax_scale
             layernorm_gamma_beta_backward per_channel_moments
optimizer    adadelta adamw momentum_sgd
contraction  attention_decode gemm gemm_silu pairwise_sqdist
other        gemm_bias
```

## Model

`deepseek-v4-flash`. Two names on this endpoint cannot pass the model-identity gate:

* `kimi-k3` is answered by `kimi-k3-fireworks` for a share of its messages, and the backend
  changes inside a single conversation (F-2026-09-17-006).
* `deepseek-v4.1-flash` is an alias answered by `deepseek-v4-1-flash-260910`, so the name
  asked for and the name that answers never agree -- it fails every time, not sometimes.

`deepseek-v4-flash` and `deepseek-v4-pro` both answer under the name they were asked for.
Check a new name before a long sweep:

```bash
bash /tmp/probe_init.sh <model>     # prints the init model and every message's model
```

## What has to be true before a run

* `~/.claude/settings.json` on bw1100 carries `ANTHROPIC_AUTH_TOKEN` (mode 600). The Lab
  strips that name from the environment, so the CLI has to hold it (F-2026-09-17-007).
  `run_dcu_task.sh` refuses at second zero if it is missing, rather than failing forty
  seconds later inside the jail as "Not logged in".
* `/home/testuser01/oci-dcu` is a clean checkout at a commit. Source identity is the
  commit (ADR 0065); a dirty checkout has no identity and preflight refuses.
* The proxy is the default for the host's shells. Direct egress to the provider is refused
  from bw1100; `mirrors.aliyun.com` is in `no_proxy` because the container's apt reaches it
  directly and breaks through the proxy.

## What a completed run looks like

```
=== exit 0 ===
run_terminal  protocol_adherence: adhered
              endpoint: best_confirmed_latency_ms 0.005439, qualified_by_budget true
```

`Task performance ... "missing": ["target_memory_bandwidth_reference"]` is expected: gfx938
has no bandwidth calibration, so the efficiency score is reported unavailable rather than
estimated from another target's number.
