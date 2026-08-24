ARM={{ARM}}
CANDIDATE_PATH_JSON={{CANDIDATE_PATH_JSON}}
This is a zero-GPU provider capability qualification for the canonical candidate-set envelope.
{{TOOL_INSTRUCTION}}
Use one file change to {{EXPECTED_CHANGE}} the candidate path above. Do not create another file.
The envelope must contain exactly {{MAXIMUM_CANDIDATES_PER_TURN}} members and must be this JSON object:
EXPECTED_CANDIDATE_SET_JSON={{EXPECTED_CANDIDATE_SET_JSON}}
Serialize it with recursively sorted keys, UTF-8, no spaces, and exactly one final newline.
Read the embedded frozen reference bundle and confirm that its nonce occurs in every member.
REFERENCE_BUNDLE_SHA256={{REFERENCE_BUNDLE_SHA256}}
{{REFERENCE_BUNDLE}}
Do not mutate an external system, invoke a GPU, use the network, or run a compiler.
After sealing the envelope, return only the requested output-schema object.
