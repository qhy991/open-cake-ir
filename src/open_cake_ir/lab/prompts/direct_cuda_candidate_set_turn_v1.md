You are authoring an ordered set of direct CUDA sources for frozen Run {{RUN_ID}}, arm {{ARM}}, Turn {{TURN}}.
Write one candidate-set envelope to {{CANDIDATE_PATH}}. The first Turn adds it; every resumed Turn updates that same
file. The envelope is `{"arm":"direct_cuda","candidates":[...],"schema_version":1}` with one to
{{MAXIMUM_CANDIDATES_PER_TURN}} non-empty CUDA source strings in provider order. Serialize the complete envelope as
canonical JSON: recursively sorted keys, UTF-8, no spaces, and exactly one final newline. Make the members
structurally distinct; renaming a kernel or reformatting source is not a new program.

Use only the direct CUDA source tool surface and the complete frozen reference bundle embedded later in this prompt.
No reference files are mounted in the writable workspace. Do not run a candidate, invoke GPU tools, use the network,
or access the Open Cake Compiler. Cumulative provider tokens before this Turn:
{{CUMULATIVE_PROVIDER_TOKENS}}. Start every member from the embedded `candidate-skeleton.cu`; preserve the Workload
shapes and first-line launch ABI unless bounded feedback requires a semantics-preserving repair. The bounded feedback
from the preceding Turn is: {{FEEDBACK_JSON}}
The complete read-only authority for this arm is embedded below:
{{REFERENCE_BUNDLE}}
After the single envelope file change, return only the output-schema object requested by the provider invocation.
