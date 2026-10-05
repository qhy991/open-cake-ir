"""Synthetic native rollout for qualification software tests, never a live CLI."""
import json
import os
from pathlib import Path


def append_rollout(thread_id, arguments):
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
        rows.append(frame('<skills_instructions>\n## Skills\n### Skill roots\n'
            f'- `r0` = `{skill.parent.parent}`\n### Available skills\n'
            '- cake: fixture (file: r0/cake/SKILL.md)\n</skills_instructions>',
            'host_skills.instructions', 'developer'))
    rows.append(row('turn_context', turn_id=native_turn, cwd=str(Path.cwd()), model=model, effort=effort))
    rows.append(frame(f'<skill>\n<name>cake</name>\n<path>{skill}</path>\n{skill.read_text()}\n</skill>',
                      'skills.selected_skill_instructions', 'user'))
    rows.append(row('event_msg', type='task_complete', turn_id=native_turn))
    with path.open('ab' if resumed else 'xb') as stream:
        stream.write(b''.join(json.dumps(item).encode() + b'\n' for item in rows))
