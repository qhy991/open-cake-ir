You are optimizing one Open Cake Schedule for frozen Run {{RUN_ID}}, arm {{ARM}}, Turn {{TURN}}.
The fixed Candidate submission Interface is {{CANDIDATE_PATH}}: create it on the first Turn and update it on every
resumed Turn. Provider-default features are available, including auxiliary shell, Apps/MCP, browser, plugins and
subagents when the pinned provider actually offers them. You may use scratch files inside the writable workspace;
only the fixed Candidate path is sealed and evaluated.

Feature availability does not authorize external mutation or direct GPU measurement. Treat auxiliary sources as
authoring priors, never as correctness or performance evidence. Only the Lab's subsequent common Evaluation Receipt
is canonical. Start from the embedded `schedule-skeleton.json`, preserve the Workload binding and external tensor
shapes, and never put credentials or private tool output in the Candidate.

Cumulative provider tokens before this Turn: {{CUMULATIVE_PROVIDER_TOKENS}}.
The bounded canonical feedback from the preceding Turn is: {{FEEDBACK_JSON}}
The complete read-only authority for this arm is embedded below:
{{REFERENCE_BUNDLE}}
After sealing the Candidate, return only the output-schema object requested by the provider invocation.
