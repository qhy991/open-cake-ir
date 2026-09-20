#!/usr/bin/env python3
"""Qualify two actual message-only Responses turns; no compiler or GPU invocation.

This command makes paid API requests when explicitly invoked with OPENAI_API_KEY.
Every request/response is retained outside source before producing a qualification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from open_cake_ir.lab.message_provider import CONTRACT, ResponsesHTTPTransport, qualify
from open_cake_ir.lab.custody import admit_new_campaign_path
from open_cake_ir.serialization import canonical_json_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='exact reported model snapshot; aliases that resolve differently are refused')
    parser.add_argument('--reasoning-effort', required=True)
    parser.add_argument('--max-output-tokens', type=int, default=2048)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--fixture-only', action='store_true',help='label controlled protocol-test evidence as non-live')
    args = parser.parse_args(argv)
    root = admit_new_campaign_path(ROOT,args.evidence_root,role='message qualification evidence')
    root.mkdir(parents=True,exist_ok=False)
    config = {'harness':'responses','model':args.model,'reasoning_effort':args.reasoning_effort,
              'max_output_tokens':args.max_output_tokens,'timeout_seconds':args.timeout_seconds,
              'sandbox':'messages_only','event_contract':CONTRACT}
    transport = ResponsesHTTPTransport(args.timeout_seconds)
    def observe(turn, role, payload):
        with (root/f'turn-{turn}.{role}.json').open('xb') as stream:
            stream.write(payload)
    receipt = qualify(config,transport,fixture=args.fixture_only,observe=observe)
    with (root/'qualification.json').open('xb') as stream: stream.write(canonical_json_bytes(receipt.document))
    print(json.dumps({'qualification':str(root/'qualification.json'),'scope':receipt.scope,
                      'reported_model':receipt.reported_model},ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
