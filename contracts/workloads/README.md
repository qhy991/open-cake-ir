# Workload Contract ownership

`flash-kmeans-assign-v2.json` is the current Flash-KMeans authority because it adds the retained b32 materialization.
`flash-kmeans-assign.json` is the historical v1 input boundary required to replay r37/r39 identities; new Studies
must not reference it. `tinygemm2-stage4-v2.json` is the current TinyGEMM2 authority because it pins input, oracle
and parent-output bytes. `tinygemm2-stage4.json` is immutable historical v1 and must not be used by new Studies.
