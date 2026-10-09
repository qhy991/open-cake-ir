"""Synthetic native rollout for qualification software tests, never a live CLI."""
import json
import os
from pathlib import Path


def append_rollout(thread_id, arguments, *, load_body=True, extra_skill=None):
    resumed = arguments[:2] == ['exec', 'resume']
    home, codex = Path(os.environ['HOME']), Path(os.environ['CODEX_HOME'])
    skill = home/'.agents/skills/cake/SKILL.md'
    native_turn = ('2' if resumed else '1') + thread_id[1:]
    path = codex/'sessions/2026/01/01'/f'rollout-fixture-{thread_id}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    model = arguments[arguments.index('--model') + 1]
    effort = next(json.loads(item.split('=', 1)[1]) for item in arguments
                  if item.startswith('model_reasoning_effort='))
    def row(kind, **payload):
        return {'type': kind, 'payload': payload}
    def frame(text, kind, role):
        return row('response_item', type='message', role=role,
            content=[{'type': 'input_text', 'text': text}],
            internal_chat_message_metadata_passthrough={
                'turn_id': native_turn, 'content_item_kinds': [kind]})
    rows = [] if resumed else [row('session_meta', id=thread_id, cwd=str(Path.cwd()), cli_version='0.159.2')]
    rows.append(row('event_msg', type='task_started', turn_id=native_turn))
    if not resumed:
        instructions = ('<skills_instructions>\n## Skills\n### Skill roots\n'
            f'- `r0` = `{skill.parent.parent}`\n### Available skills\n'
            '- cake: fixture (file: r0/cake/SKILL.md)\n'
            + (f'- {extra_skill[0]}: fixture (file: r0/{extra_skill[1]}/SKILL.md)\n' if extra_skill else '')
            + '</skills_instructions>')
        rows.append(frame(instructions, 'host_skills.instructions', 'developer'))
        rows.append(row('world_state', full=True, state={'host_skills': {
            'body': instructions.removeprefix('<skills_instructions>').removesuffix('</skills_instructions>'),
            'includeInstructions': True}}))
    rows.append(row('turn_context', turn_id=native_turn, cwd=str(Path.cwd()), model=model, effort=effort))
    if load_body:
        rows.append(frame(f'<skill>\n<name>cake</name>\n<path>{skill}</path>\n{skill.read_text()}\n</skill>',
                          'skills.selected_skill_instructions', 'user'))
    if extra_skill and extra_skill[2]:
        name, directory, _ = extra_skill
        extra_path = skill.parent.parent/directory/'SKILL.md'
        rows.append(frame(f'<skill>\n<name>{name}</name>\n<path>{extra_path}</path>\n{extra_path.read_text()}\n</skill>',
                          'skills.selected_skill_instructions', 'user'))
    rows.append(row('event_msg', type='task_complete', turn_id=native_turn))
    with path.open('ab' if resumed else 'xb') as stream:
        stream.write(b''.join(json.dumps(item).encode() + b'\n' for item in rows))


def emit_runtime_turn(arguments, projection):
    """CPU executable fixture using existing candidate/events and native facts owners.

    No model or GPU is invoked. The real adapter observes these synthetic events and
    files; the test must never publish their synthetic qualification as a real one.
    """
    from types import SimpleNamespace
    from open_cake_ir.lab.task_package import TaskPackage
    from tests.contracts.test_lab import RalphFakeProvider

    resumed = arguments[:2] == ['exec', 'resume']
    task = TaskPackage(projection['run_id'], projection['arm'],
                       projection['task_markdown'], projection['agents_markdown'])
    state = projection['state_card']
    request = SimpleNamespace(run_id=task.run_id, arm=task.arm, turn=state['iteration'],
        thread_id=arguments[-2] if resumed else None,
        cumulative_provider_tokens=state['cumulative_provider_tokens'], state_card=state)
    result = RalphFakeProvider({task.run_id: task}).turn(request)
    candidate = Path.cwd()/'candidate-set.json'
    candidate.write_bytes(result.raw_submission)
    append_rollout(result.thread_id, arguments)
    # The reusable CPU provider uses /fixture as its synthetic event path. This
    # executable owns a real task directory, so retain the actual changed file.
    for raw in result.raw_events.splitlines():
        event = json.loads(raw)
        item = event.get('item', {})
        if item.get('type') == 'file_change':
            for change in item['changes']:
                change['path'] = str(candidate)
        print(json.dumps(event, separators=(',', ':')))
