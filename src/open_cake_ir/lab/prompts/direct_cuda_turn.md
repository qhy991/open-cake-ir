You are authoring one direct CUDA candidate for frozen Run {{RUN_ID}}, arm {{ARM}}, Turn {{TURN}}.
Write exactly one candidate to {{CANDIDATE_PATH}}. The first Turn adds it; every resumed Turn updates that same file.
Use only the direct CUDA source tool surface and the frozen references already mounted in this workspace. Do not run
the candidate, invoke GPU tools, use the network, or access the Open Cake Compiler. Cumulative provider tokens before
this Turn: {{CUMULATIVE_PROVIDER_TOKENS}}. Start from the complete `candidate-skeleton.cu`; preserve its Workload
shapes and first-line launch ABI unless bounded feedback requires a semantics-preserving repair. The bounded feedback
from the preceding Turn is: {{FEEDBACK_JSON}}
The complete read-only authority for this arm is embedded below:
{{REFERENCE_BUNDLE}}
After the single file change, return only the output-schema object requested by the provider invocation.
