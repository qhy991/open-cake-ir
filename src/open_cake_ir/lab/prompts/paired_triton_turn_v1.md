Optimize the supplied known baseline for the frozen Workload case. Keep its explicit
Triton backend, tensor ABI, oracle and treatment assignment. Host Python is forbidden.
The Open Cake arm submits a Schedule object or {"python_source": "restricted IR source"};
the native_triton arm submits {"kernel_source": "...", "compile_constants": {...},
"compile_options": {...}, "grid": [x,y,z]}. The embedded authoring contract defines the
supported syntax. Native source is one kernel with fixed imports and @triton.jit only;
its trusted launcher is outside candidate control. Both arms begin with the exact same
Compiler-generated kernel implementation. Source-only preparation is not GPU equality.

Run: {{RUN_ID}}
Arm: {{ARM}}
Turn: {{TURN}}
Write exactly one canonical candidate-set envelope at: {{CANDIDATE_PATH}}
Cumulative provider tokens: {{CUMULATIVE_PROVIDER_TOKENS}}
Maximum structurally distinct candidates: {{MAXIMUM_CANDIDATES_PER_TURN}}
Feedback: {{FEEDBACK_JSON}}
Frozen reference bundle:
{{REFERENCE_BUNDLE}}

Use only the assigned candidate file and output the schema-required terminal record.
Do not invoke a GPU, install software, alter the oracle, or declare a measured result.
