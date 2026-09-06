# Third-party dependencies and provenance

The Apache-2.0 license applies to project-authored code and documentation. It does not
relicense an external paper, library, dataset, or tool. Preserve any third-party notices
when redistributing material from those sources.

## Dependencies and references

The following projects are used as dependencies, compiler backends, or reference sources.
This table is not a statement that their complete distributions are bundled here. Consult
the exact dependency release and file notices when distributing a combined package.

| Project | Relationship | Upstream license information |
| --- | --- | --- |
| Triton | Generated kernel source and optional GPU compiler/runtime | [MIT-style license](https://github.com/triton-lang/triton/blob/main/LICENSE) |
| CUTLASS / CuTe DSL | Optional generated backend and toolchain | [CUTLASS BSD-3-Clause](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt); inspect the actual DSL distribution separately |
| PyTorch | Optional reference/evaluation and GPU environment | [PyTorch license and contributor notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| jsonschema | Schema and contract tests | [MIT license](https://github.com/python-jsonschema/jsonschema/blob/main/COPYING) |
| FlashInfer | Fixed upstream workload/reference provenance | [Apache-2.0 and listed third-party components](https://github.com/flashinfer-ai/flashinfer/blob/main/LICENSE) |
| CAKE paper | Research design reference | [Paper and its own license](https://arxiv.org/abs/2608.12629v1) |

CUDA, Nsight Compute, and Compute Sanitizer are external tools. Install them through their
respective vendors under the applicable terms; this repository does not distribute their
tool installations.

## Project history and retained evidence

The `cake-repro` source lineage is retained in `migration/` for this project's historical
replay. Generated templates identify their original project generator. Project-authored
material in that lineage is included in this project's Apache-2.0 grant; external
references and any third-party notices keep their original terms.

The `evidence/` and `docs/data/` trees contain historical records, generated artifacts, and
projections with their own source provenance. A measurement record is not a license grant
for an external model, source repository, or dataset. In particular, AKA source locators
identify the inputs used for a review; they do not distribute or relicense the complete
AKA dataset. Generated compiler outputs retain any embedded third-party notices.

Do not copy a dependency's full source or binary distribution into a contribution without
including its required license and attribution. When a contribution adds third-party
material, identify the exact source, version, affected files, and redistribution terms.
