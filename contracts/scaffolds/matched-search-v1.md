# Matched-search scaffold v1

Each Run receives one empty writable candidate workspace and an immutable external reference bundle whose bytes are
embedded in each Turn prompt. The bundle contains its Workload Contract, Study/Campaign authority, task statement,
candidate skeleton and assigned Authoring Environment. The provider may change only the one
candidate file named by the Turn prompt. It cannot invoke the Compiler, toolchain, evaluator, GPU, network or the
other treatment. Feedback is limited to the Study-declared projection from the preceding Turn.
