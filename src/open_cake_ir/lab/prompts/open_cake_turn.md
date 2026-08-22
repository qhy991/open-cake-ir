You are authoring one Open Cake Schedule for frozen Run {{RUN_ID}}, arm {{ARM}}, Turn {{TURN}}.
Write exactly one candidate to {{CANDIDATE_PATH}}. The first Turn adds it; every resumed Turn updates that same file.
Use only the Schedule tool surface and the frozen references already mounted in this workspace. Do not invoke CUDA,
GPU tools, network tools, or another compiler. Start from the complete `schedule-skeleton.json` and preserve its
profile, Workload binding and external tensor shapes. Cumulative provider tokens before this Turn: {{CUMULATIVE_PROVIDER_TOKENS}}.
The bounded feedback from the preceding Turn is: {{FEEDBACK_JSON}}
The complete read-only authority for this arm is embedded below:
{{REFERENCE_BUNDLE}}
After the single file change, return only the output-schema object requested by the provider invocation.
