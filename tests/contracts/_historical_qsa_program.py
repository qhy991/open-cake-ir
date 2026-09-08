"""Replay Program v2 with its pinned historical Compiler in a fresh process.

This is a source snapshot from Git, not a copy of a legacy runner into current
implementation. The current Compiler must continue to reject the old Program.
"""
from __future__ import annotations

from functools import lru_cache
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROGRAM_V2_COMMIT = "a302ee396ce4f130692f2fcca0243d357f9659f2"


@lru_cache(maxsize=1)
def replay_program_v2() -> dict:
    present = subprocess.run(["git", "cat-file", "-e", PROGRAM_V2_COMMIT + "^{commit}"],
                             cwd=ROOT, capture_output=True)
    if present.returncode:
        raise unittest.SkipTest("Program v2 replay requires pinned historical Git a302ee3")
    archive = subprocess.check_output(["git", "archive", PROGRAM_V2_COMMIT,
        "src", "compiler", "corpus", "contracts", "docs/PYTHON_FRONTEND.md",
        "examples/python", "examples/gpu"], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="cake-program-v2-replay-") as directory:
        root = Path(directory).resolve()
        with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
            # Only this trusted Git tree's ordinary files/directories are needed.
            if any(not (member.isfile() or member.isdir()) or
                   Path(member.name).is_absolute() or ".." in Path(member.name).parts
                   for member in snapshot.getmembers()):
                raise ValueError("historical Program snapshot contains a non-source member")
            snapshot.extractall(root)
        completed = subprocess.run([sys.executable, "-I", "-c", _REPLAY], cwd=root,
                                   capture_output=True, text=True, check=True)
        return json.loads(completed.stdout)


_REPLAY = r'''
import json,sys
from pathlib import Path
root=Path.cwd().resolve()
sys.path.insert(0,str(root/'src'))
import open_cake_ir.compiler as compiler_module
import open_cake_ir.tasks.qsa.program as program_module
for module in (compiler_module,program_module):
    assert Path(module.__file__).resolve().is_relative_to(root/'src')
compiler=compiler_module.Compiler.load(root,root/'compiler/revision.lock.json')
path=root/'contracts/programs/qsa-prefill-t32768-v2.json'
original=json.loads(path.read_text())
program=program_module.ProgramContract.load(root,path,compiler)
assessment=compiler.assess_file(root/'corpus/schedules/qsa-score-topk-t32768.json')
result={'program_id':program.program_id,'public_outputs':list(program.public_outputs),
        'nodes':[node.node_id for node in program.nodes],
        'last_dependencies':list(program.nodes[-1].depends_on),
        'reference_visibility':program.reference_visibility,
        'qsa_lowering_source_sha256':compiler.lower(assessment).source_sha256,
        'mutation_refusals':[]}
for index in range(3):
    document=json.loads(json.dumps(original))
    if index==0: document['nodes'][0]['schedule_sha256']='0'*64
    elif index==1: document['nodes'][1]['depends_on']=['ghost']
    else: document['nodes'][-1]['views']['output']='identity'
    mutation=root/f'program-mutation-{index}.json'
    mutation.write_text(json.dumps(document))
    try: program_module.ProgramContract.load(root,mutation,compiler)
    except ValueError as error: result['mutation_refusals'].append(str(error))
    else: raise AssertionError('historical Program mutation did not refuse')
print(json.dumps(result))
'''
