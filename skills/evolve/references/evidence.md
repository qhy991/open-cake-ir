# Reading development evidence

The current external adapter supports the retained allocation format
`bw1100.round5.static-handoff.v1`. The allocation owns task membership, host/root,
Compiler, model, reference access, cases and budget. It does not own task outcomes.
For this handoff format, assignment rows retain the original scaffold gateway;
`hosts[host].gateway` owns the destination runtime selection used by preparation.
The projection retains `source_gateway` separately and checks the effective gateway.

Run the reader on a machine that can read that host's assigned roots:

```bash
python3 tools/evolve.py inspect-development \
  --allocation /research/round/allocation.json --host worker-a
```

Use the actual path and host key from the allocation. The command writes JSON only to
stdout. Capture it outside the source checkout. It reads `development-binding.json`,
`ENDPOINT.json`, `DONE.json`, and retained evaluation outcomes/assessments. It checks
membership, controls and final case coverage, then lists source and profiler locators
for review. It does not execute retained Python, load tensors, replay the Compiler,
read credentials, call a Provider, acquire devices, or modify experiment files.

For a local snapshot retaining absolute path layout, add `--mirror /research/snapshot`.
For example, a remote root `/data/round/runs/task` is then read from
`/research/snapshot/data/round/runs/task`. Source locators continue to point to the
original host. A snapshot's result is not a live host observation.

Join host reports against one copy of the original allocation:

```bash
python3 tools/evolve.py merge-development \
  --allocation /research/round/allocation.json \
  /research/review/worker-a.json /research/review/worker-b.json
```

No observations are silently dropped. Duplicate tasks, changed controls or wrong-host
rows are refused. Missing hosts appear as `unobserved`. Task states mean:

| State | Meaning |
| --- | --- |
| `confirmed` | Retained owner records agree on terminal confirmation, all assigned cases and release. |
| `failed` | The owner recorded a terminal failed/unqualified result and release. |
| `incomplete` | No terminal record; inspect the existing launch owner before acting. |
| `release_pending` | Terminal records do not establish release. |
| `invalid` | Missing, malformed or conflicting evidence prevents classification. |
| `unobserved` | No report supplied for this assigned task. |

`ready_for_maintenance_review` means every assigned task has a classifiable terminal
record with reported release. It does not establish semantic replay, present device
availability, a speedup, or permission to dispatch. Early review of sealed individual
tasks is allowed while other tasks run; the full discovery phase remains incomplete.

All accepted and rejected evaluation directories are retained in the review input.
Missing outcome files remain visible. Read the emitted source and profiler reports to
distinguish a useful kernel from a reusable Compiler mechanism. Read release receipts
with the gateway's own verifier before subsequent device admission; the reader reports
the retained owner's release verdict rather than reimplementing that verifier.
Candidate names do not establish whether the original starter was retained. Compare
the nominee's retained source with the canonical starter and read the owner's rationale.

For native EvidenceStore Runs, retain their existing reader and pinned semantic audit:

```bash
python3 tools/summarize_diagnoses.py /research/native-run/evidence --compiler-gaps
```

Use the producer-compatible checkout. An external development projection never becomes
a native Run authority merely because it has been read successfully.
