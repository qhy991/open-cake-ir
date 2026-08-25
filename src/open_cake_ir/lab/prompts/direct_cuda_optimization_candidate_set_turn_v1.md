You are optimizing an ordered set of direct CUDA sources for frozen Run {{RUN_ID}}, arm {{ARM}}, Turn {{TURN}}.
The fixed Candidate submission Interface is {{CANDIDATE_PATH}}: create it on the first Turn and update it on every
resumed Turn. It is one `{"arm":"direct_cuda","candidates":[...],"schema_version":1}` envelope containing one to
{{MAXIMUM_CANDIDATES_PER_TURN}} structurally distinct non-empty CUDA source strings in provider order. Serialize the
complete envelope as canonical JSON: recursively sorted keys, UTF-8, no spaces, and exactly one final newline.
Renaming a kernel or reformatting source is not a new program.

Provider-default features are available, including auxiliary shell, Apps/MCP, browser, plugins and subagents when the
pinned provider actually offers them. Use auxiliary agents only for read-only, distinct investigations; this primary
thread alone owns the envelope write. Leave no other workspace file at the Turn boundary. Feature availability does
not authorize external mutation or direct GPU measurement. Treat auxiliary sources as authoring priors, never as
correctness or performance evidence. Only the Lab's subsequent common Evaluation Receipt is canonical. Start every member from the embedded
`candidate-skeleton.cu`, preserve the Workload shapes and first-line launch ABI, and never put credentials or private
tool output in the envelope.

Cumulative provider tokens before this Turn: {{CUMULATIVE_PROVIDER_TOKENS}}.
The bounded canonical feedback from the preceding Turn is: {{FEEDBACK_JSON}}
The complete read-only authority for this arm is embedded below:
{{REFERENCE_BUNDLE}}
After sealing the envelope, return only the output-schema object requested by the provider invocation.
