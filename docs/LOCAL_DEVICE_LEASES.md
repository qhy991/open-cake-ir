# Local device leases

The existing local broker remains the allocation owner. The default `user` lock scope
preserves frozen workers. `--local-lock-scope device` requires an explicit
`--local-device` and uses the physical ordinal from the host runtime namespace.
In a MACA container exposing only part of the host, runtime ordinals can differ.
Declare all three facts: `--local-device` is the host physical lock key,
`--local-runtime-device` is the container's unfiltered ordinal, and
`--local-expected-pci` is its host PCI address (`dddd:bb:dd`). The mapping requires
device scope. Admission compares the actual native PCI address before any kernel
launch. Missing, partial or mismatched mappings refuse. Runtime visibility never
changes the host lock key. Retain the host inventory and probe with the Run inputs.

The worker must prove the intended visible device and native PCI identity before
this becomes a qualified hardware route. A lock name alone does not prove mapping.

## Compatibility and lifetime

A device-scoped job takes the existing user/backend lock in shared mode, then its
own device lock in exclusive mode. Different device-scoped jobs can overlap;
the same device cannot. Existing workers take the legacy lock exclusively and
therefore exclude all device-scoped work while their allocation lasts. The new
scope does not bypass or unlink a legacy lock, and does not claim simultaneous
legacy/new device execution. Old and new authoring processes can still run while
neither owns a device lease.

A failed acquisition releases every acquired descriptor before waiting. Both
successful descriptors survive exec and remain held until the short-lived device
worker exits. Host inputs and oracle preparation remain before allocation. The
same baseline/candidate pair stays inside one complete device phase.

This scope is opt-in and rides the frozen runtime command. It does not change a
running Run, its source, its budget or its device. A new source/Executor and its
applicable CPU, mapping, correctness, timing and profiler gates are required before
formal parallel experiments. `local_serialized` continues to mean cooperative
serialization of the admitted local device, not physical-node exclusivity.

## Bounded parallel experiments

Use independent canonical task-launcher processes and immutable per-Run workspaces.
Fix task-to-device assignments before search. Compare each candidate with its own
fixed baseline on that same physical card; retain PCI identity in device evidence.
Use the same concurrency and assignments for the C0/C1 comparison, or report the
change separately. Provider work and compilation do not reserve a device.

The intended first C550 successor uses GPUs1-4 for four author/search workers and
keeps the frozen pilot assigned to GPU0. Legacy compatibility may briefly queue
measurements; it does not require waiting for the entire old Study to finish.
After mapping, serial A/A and simultaneous A/A/profiler controls pass, this can be
qualified as four device-scoped measurement slots. This document is not evidence
that those device controls have passed.

## Metal local evaluation phases

Metal uses the existing user-scoped single-device lock; device-ordinal selection is
not supported. Its local launcher starts `tasks.evaluate --local-kind metal` without
a lease. The task prepares every required input and original CPU reference, including
all validation distributions for attribution. A short-lived broker child then execs
the already admitted native observer for the entire declared plan. Paired candidate
and baseline cohorts, warmups, snapshots and required synchronization remain inside
that one device interval. The parent compares retained snapshots and writes reports
after the child exits and is reaped. Release does not mean correctness acceptance.

The same process runner admits archive build/reload and host inspection separately;
Metal archive construction itself touches the device and therefore needs a lease.
An inherited allocation retains its existing owner and lifetime. Neither the parent
nor reporting code unlocks another process's allocation. Retained admission identifies
the actual job, including an unsuccessful broker attempt; it does not establish
physical exclusion of other applications. CPU regression tests prove process and
oracle ordering only. A successor still needs native host/device acceptance before
this route is used by formal performance experiments.
