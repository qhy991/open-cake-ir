# Python source-file Provider submission

Status: accepted for one-candidate Python Authoring Environments; no live Provider
qualification or GPU result is claimed by software fixtures.

New `python_source_file_v1` Runs let one Cake author write one `candidate.py` per Turn.
The Provider adapter witnesses the add/update lifecycle, reads the regular file without
following symlinks, and retains its exact UTF-8 bytes. A deterministic projection
creates the internal `{"python_source": ...}` Candidate passed to the existing Lab and
Compiler. Archive and replay retain and independently reproject the raw source file;
the author never writes a Schedule JSON or candidate-set envelope in this treatment.

Admission requires `open_cake`, `python_source_v1`, known-kernel reproduction, one
candidate and one search per Turn, and no granted transform. The Provider executable,
configuration and exact submission contract require their own two-Turn qualification;
an envelope-v1 receipt cannot qualify source-file submission. Both Codex and Claude
adapters use the same frozen file name and lifecycle. Old envelope-v1 Runs, message
providers, direct CUDA, native comparisons, multi-candidate Turns and transform actions
keep their original contracts and replay at their pinned commits.

The remaining migration is not a file rename. Multiple `.py` files need a witnessed
ordering and complete custody record, and transform actions need a separate explicit
author surface. Their successor contracts must preserve the current candidate and
action ordering, raw bytes, rejection attribution and semantic replay before the
envelope treatment can be retired for new work.
