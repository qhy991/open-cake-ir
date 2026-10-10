"""Compiler author tools for external hmz workspaces; no search or Evaluation owner."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import time

CATALOG = 'campaign/compiler-api.json'
ACTIONS = 'campaign/compiler-actions'


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def inside(root, name):
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def compiler_pin(root):
    bench, development = root / 'campaign/binding.json', root / 'campaign/development-binding.json'
    if bench.exists() == development.exists():
        raise ValueError('Expected exactly one Bench or development binding')
    return read(bench)['compiler_commit'] if bench.exists() else read(development)['compiler']


def compiler(root):
    source = (root / '.deps/cake-ir').resolve()
    loaded = sys.modules.get('open_cake_ir')
    if loaded is not None and Path(loaded.__file__).resolve().parent != source / 'src/open_cake_ir':
        raise ValueError('Another Compiler is already imported; use a fresh process for each condition')
    sys.path.insert(0, str(source / 'src'))
    from open_cake_ir.compiler import Compiler
    engine = Compiler.load(source, source / 'compiler/revision.json')
    if engine.commit != compiler_pin(root):
        raise ValueError('Compiler must be the clean pinned source')
    return engine


def freeze(root):
    if (root / 'campaign/deadline.json').exists():
        raise ValueError('Prepare the API before Run intake; never retrofit an active Run')
    engine = compiler(root)
    from open_cake_ir.compiler.program_passes import TRANSFORMATIONS
    from open_cake_ir.lab.knowledge import transformation_surface
    document = dict(compiler_commit=engine.commit,
                    transformations=transformation_surface([item.name for item in TRANSFORMATIONS]))
    write_new(root / CATALOG, document)
    return document


def catalog(root):
    path = inside(root, CATALOG)
    # Existing intake owns the prepared source identity. Before intake, use HEAD.
    owner = root / ('campaign/development-binding.json')
    intake = root / ('campaign/deadline.json' if owner.exists() else 'campaign/intake.json')
    revision = read(intake)['source_commit'] if intake.exists() else 'HEAD'
    frozen = subprocess.check_output(['git', '-C', str(root), 'show', revision + ':' + CATALOG])
    if path.read_bytes() != frozen:
        raise ValueError('Frozen Compiler API changed')
    value = json.loads(frozen)
    if value['compiler_commit'] != compiler_pin(root):
        raise ValueError('Compiler API belongs to a different condition')
    return value


def history(root):
    rows = []
    for path in (root / ACTIONS).glob('*/request.json'):
        try:
            request = read(path)
            result_path = path.with_name('result.json')
            result = read(result_path) if result_path.exists() else {'reason': 'unknown_incomplete'}
            rows.append(dict(id=path.parent.name, started_at_epoch=request['started_at_epoch'],
                             transformation=request['action'].get('transformation'),
                             parent=request['action'].get('parent'), reason=result['reason'],
                             message=result.get('message'),
                             record=str(result_path.relative_to(root))))
        except (OSError, ValueError, KeyError, TypeError):
            rows.append(dict(id=path.parent.name, started_at_epoch=0, reason='unknown_unreadable'))
    rows.sort(key=lambda row: (row['started_at_epoch'], row['id']))
    return dict(actions=rows[-20:], omitted=max(0, len(rows) - 20))


def author_context(root):
    return ('\nFrozen Compiler transformation API. This catalog describes public tools, not measured gains.\n'
            + json.dumps(catalog(root), ensure_ascii=False, indent=2)
            + '\nOwn transform observations; correctness and performance require the existing evaluator.\n'
            + json.dumps(history(root), ensure_ascii=False, indent=2))


def parent_source(root, name):
    path = inside(root, name)
    relative = path.relative_to(root.resolve())
    if (len(relative.parts) < 3 or relative.parts[:2] not in {
            ('campaign', 'candidates'), ('campaign', 'schedules'),
            ('campaign', 'inherited'), ('campaign', 'compiler-actions')}
            or path.suffix not in ('.py', '.json')):
        raise ValueError('Parent must be an own candidate/Schedule, declared inherited source, or prior tool result')
    if relative.parts[1] == 'inherited' and not (root / 'campaign/development-binding.json').exists():
        raise ValueError('Independent Bench has no inherited development parent')
    if relative.parts[1] == 'compiler-actions':
        if len(relative.parts) != 4 or not (path.name == 'program.json' or re.fullmatch(r'stage-[0-9]{3}\.json', path.name)):
            raise ValueError('A tool parent must be its retained Program or stage Schedule')
        verify(root, relative.parts[2])
    source = path.read_bytes()
    return json.dumps({'python_source': source.decode()}).encode() if path.suffix == '.py' else source


def inspect_parent(root, name):
    compiler(root)
    from open_cake_ir.lab.actions import candidate_program
    program = candidate_program(parent_source(root, name))
    return dict(parent=name, stages=[dict(name=stage.name, schedule_id=stage.schedule.schedule_id,
                                        entry_point=stage.schedule.lowering.entry_point)
                                   for stage in program.stages])


def action_directory(root, identity):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', identity):
        raise ValueError('Use a unique lowercase action id')
    return inside(root, ACTIONS + '/' + identity)


def stage_origin(root, name, document):
    """Bind a tool-generated Schedule to the existing emitter's source record."""
    path = inside(root, name)
    relative = path.relative_to(root.resolve())
    if relative.parts[:2] != ('campaign', 'compiler-actions'):
        return None
    if len(relative.parts) != 4 or not re.fullmatch(r'stage-[0-9]{3}\.json', path.name):
        raise ValueError('Emit one complete stage Schedule from the tool result')
    verify(root, relative.parts[2])
    if read(path) != document:
        raise ValueError('Emitted Schedule differs from its declared transform result')
    return str(relative)


def resolve(engine, api, action, parent):
    from open_cake_ir.lab.actions import resolve_action
    return resolve_action(json.dumps(action).encode(), environment_kind='open_cake',
                          transformations=[item['name'] for item in api['transformations']],
                          candidates={action['parent']: parent}, baselines={},
                          compiler_factory=lambda: engine, allow_python=True)


def transform(root, identity, action):
    api = catalog(root)
    deadline = read(root / 'campaign/deadline.json')
    if time.time() >= deadline['search_stop_at_epoch']:
        raise ValueError('Search is closed; no new transform action')
    closed = root / 'campaign/search-close-owner.json'
    if ((root / 'campaign/development-binding.json').exists() and closed.exists()
            and read(closed).get('decision') == 'approved'):
        raise ValueError('Development owner already closed search; use read-only replay')
    if (not isinstance(action, dict) or set(action) != {'action', 'parent', 'transformation', 'parameters'}
            or action['action'] != 'transform' or not isinstance(action['parent'], str)):
        raise ValueError('Use a transform action with an own parent path, transformation and parameters')
    engine = compiler(root)
    parent = parent_source(root, action['parent'])
    folder = action_directory(root, identity)
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / 'request.json', dict(compiler_commit=engine.commit, action=action,
                                          started_at_epoch=time.time()))
    (folder / 'parent.json').write_bytes(parent)
    result = resolve(engine, api, action, parent)
    if result.candidate is not None:
        from open_cake_ir.compiler import Program
        (folder / 'program.json').write_bytes(result.candidate)
        program = Program.from_dict(json.loads(result.candidate))
        for index, stage in enumerate(program.stages):
            (folder / ('stage-%03d.json' % index)).write_bytes(stage.schedule_bytes)
    record = dict(result.document, compiler_commit=engine.commit,
                  scope='external author-tool observation; no Evaluation or native Run terminal',
                  completed_at_epoch=time.time())
    write_new(folder / 'result.json', record)
    return record


def verify(root, identity):
    """Replay only retained source/intent. Never write a candidate or spend device time."""
    engine, api = compiler(root), catalog(root)
    folder = action_directory(root, identity)
    request, record = read(folder / 'request.json'), read(folder / 'result.json')
    if request['compiler_commit'] != engine.commit or record['compiler_commit'] != engine.commit:
        raise ValueError('Transform source identity changed')
    result = resolve(engine, api, request['action'], (folder / 'parent.json').read_bytes())
    if any(record.get(key) != value for key, value in result.document.items()):
        raise ValueError('Retained transform action/result differs from replay')
    paths = sorted(folder.glob('stage-*.json'))
    if result.candidate is None:
        if (folder / 'program.json').exists() or paths:
            raise ValueError('Refused action cannot contain generated candidates')
    else:
        from open_cake_ir.compiler import Program
        if (folder / 'program.json').read_bytes() != result.candidate:
            raise ValueError('Generated Program changed')
        stages = Program.from_dict(json.loads(result.candidate)).stages
        if paths != [folder / ('stage-%03d.json' % i) for i in range(len(stages))]:
            raise ValueError('Generated stage set changed')
        if any(path.read_bytes() != stage.schedule_bytes for path, stage in zip(paths, stages)):
            raise ValueError('Generated Schedule changed')
    return dict(id=identity, replay='matched', device_qualification=False)


def main(root=None):
    root = Path.cwd().resolve() if root is None else Path(root).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('freeze')
    commands.add_parser('catalog')
    commands.add_parser('inspect').add_argument('--parent', required=True)
    action = commands.add_parser('transform')
    action.add_argument('--id', required=True)
    action.add_argument('--request', type=Path, required=True)
    commands.add_parser('verify').add_argument('--id', required=True)
    args = parser.parse_args()
    if args.command == 'freeze': result = freeze(root)
    elif args.command == 'catalog': result = catalog(root)
    elif args.command == 'inspect': result = inspect_parent(root, args.parent)
    elif args.command == 'transform': result = transform(root, args.id, read(inside(root, args.request)))
    else: result = verify(root, args.id)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
