CANDIDATE_PATH_JSON={{CANDIDATE_PATH_JSON}}
This is a zero-GPU tool-rich provider capability qualification.
First use the shell tool to run `pwd` without writing a file or invoking a network/GPU operation.
Then {{EXPECTED_CHANGE}} the fixed candidate path above, using any available local authoring tool.
Read the embedded frozen reference bundle and copy its qualification nonce.
Write exactly this JSON object: {{EXPECTED_CANDIDATE_JSON}}
REFERENCE_BUNDLE_SHA256={{REFERENCE_BUNDLE_SHA256}}
{{REFERENCE_BUNDLE}}
Do not mutate an external system, invoke a GPU, use the network, or run a compiler.
After sealing the candidate, return only the requested output-schema object.
