You are authoring an ordered set of Open Cake Schedules for frozen Run {{RUN_ID}}, arm {{ARM}}, Turn {{TURN}}.
Write one candidate-set envelope to {{CANDIDATE_PATH}}. The first Turn adds it; every resumed Turn updates that same
file. The envelope is `{"arm":"open_cake","candidates":[...],"schema_version":1}` with one to
{{MAXIMUM_CANDIDATES_PER_TURN}} Schedule objects in provider order. Serialize the complete envelope as canonical
JSON: recursively sorted keys, UTF-8, no spaces, and exactly one final newline. Make the members structurally
distinct; renaming or reordering one Schedule is not a new program.

Use only the Schedule tool surface and the complete frozen reference bundle embedded later in this prompt. No
reference files are mounted in the writable workspace. Do not invoke CUDA, GPU tools, network tools, or another
compiler. Start from the embedded `schedule-skeleton.json` and preserve its profile, Workload binding and external
tensor shapes. Cumulative provider tokens before this Turn: {{CUMULATIVE_PROVIDER_TOKENS}}.
The bounded feedback from the preceding Turn is: {{FEEDBACK_JSON}}
The complete read-only authority for this arm is embedded below:
{{REFERENCE_BUNDLE}}
After the single envelope file change, return only the output-schema object requested by the provider invocation.
