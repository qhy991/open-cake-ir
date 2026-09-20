"""A message-only Responses provider: the model never receives host tools or paths to open.

Only the frozen task bundle and this Run's native response history cross the model
boundary. The model emits JSON actions; Compiler/build/Evaluation remain in Lab.
Official protocol: https://developers.openai.com/api/docs/guides/conversation-state
and https://developers.openai.com/api/docs/guides/tools . No SDK dependency or retries.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import uuid
import urllib.error
import urllib.request

from open_cake_ir.serialization import canonical_json_bytes
from .provider_documents import (ProviderTurn, CANDIDATE_SET_ENVELOPE_V1,
    _project_candidate_submission, _unique_json_object, _reject_json_constant)
from .faults import RunProtocolFault, ReportedProviderUsage

CONTRACT = 'responses_messages_v1'
REVISION = 'open-cake-responses-messages-v1'
ENDPOINT = 'https://api.openai.com/v1/responses'
_INSTRUCTIONS = ('Use only the supplied task and same-Run history. Return one JSON candidate-set envelope '
                 'with schema_version, arm, and candidates. You have no tools, file access, web access, '
                 'or executable callbacks. Propose submit/transform actions as JSON; Lab decides and executes them.')
_CONFIG_FIELDS = {'harness', 'model', 'reasoning_effort', 'max_output_tokens', 'timeout_seconds', 'sandbox', 'event_contract'}


def configuration(provider):
    if not isinstance(provider, Mapping) or set(provider) != _CONFIG_FIELDS | {'revision', 'qualification'}:
        raise ValueError('message provider binding fields differ')
    value = {name: provider[name] for name in _CONFIG_FIELDS}
    validate_configuration(value)
    if provider['revision'] != REVISION:
        raise ValueError('message provider implementation revision differs')
    return value


def validate_configuration(value):
    if (not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS
        or value['harness'] != 'responses' or value['sandbox'] != 'messages_only'
        or value['event_contract'] != CONTRACT):
        raise ValueError('message provider configuration differs')
    for key in ('model', 'reasoning_effort'):
        if not isinstance(value[key], str) or not value[key] or value[key] != value[key].strip():
            raise ValueError(f'message provider {key} must be explicit')
    for key in ('max_output_tokens', 'timeout_seconds'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(f'message provider {key} must be positive')


def _json(value):
    return json.loads(value, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)


def request_body(config, history, bundle):
    validate_configuration(config)
    return {'model':config['model'], 'reasoning':{'effort':config['reasoning_effort']},
        'max_output_tokens':config['max_output_tokens'], 'instructions':_INSTRUCTIONS,
        'input':[ *history, {'role':'user','content':bundle.decode('utf-8')} ],
        'tools':[], 'tool_choice':'none', 'store':False, 'stream':False,
        'text':{'format':{'type':'json_object'}}}


def usage(response):
    if not isinstance(response, Mapping) or not isinstance(response.get('usage'), Mapping):
        raise ValueError('message provider usage is unavailable')
    value = response['usage']
    if any(type(value.get(key)) is not int or value[key] < 0 for key in ('input_tokens','output_tokens','total_tokens')):
        raise ValueError('message provider usage differs')
    if value['input_tokens'] + value['output_tokens'] != value['total_tokens']:
        raise ValueError('message provider total usage differs')
    return value['total_tokens']


def response_submission(response, *, expected_model=None):
    if (not isinstance(response, Mapping) or response.get('status') != 'completed'
        or not isinstance(response.get('id'), str) or not response['id']
        or not isinstance(response.get('model'), str) or not response['model']
        or expected_model is not None and response['model'] != expected_model
        or not isinstance(response.get('output'), list)):
        raise ValueError('message provider response identity or completion differs')
    if usage(response) <= 0:
        raise ValueError('completed message provider response has no observed usage')
    finals = []
    for item in response['output']:
        if not isinstance(item, Mapping):
            raise ValueError('message provider output item differs')
        if item.get('type') == 'reasoning':
            continue  # Retained intact in native history, including encrypted content.
        if (item.get('type') != 'message' or item.get('role') != 'assistant'
            or not isinstance(item.get('content'), list)):
            raise ValueError('message provider returned a tool or non-message output')
        if any(not isinstance(part, Mapping) or part.get('type') != 'output_text'
               or not isinstance(part.get('text'), str) for part in item['content']):
            raise ValueError('message provider output is not text')
        if item.get('phase') in {None, 'final'}:
            finals.append(''.join(part['text'] for part in item['content']))
    if len(finals) != 1:
        raise ValueError('message provider must return exactly one final JSON envelope')
    return finals[0].encode('utf-8')


def validate_exchange(raw, *, config, history, bundle, run_id, turn, thread_id=None, expected_model=None):
    record = _json(raw)
    if (not isinstance(record, Mapping) or set(record) != {'schema_version','kind','run_id','turn','thread_id','request','response'}
        or type(record['schema_version']) is not int or record['schema_version'] != 1
        or type(record['turn']) is not int or record['kind'] != CONTRACT
        or record['run_id'] != run_id or record['turn'] != turn
        or thread_id is not None and record['thread_id'] != thread_id):
        raise ValueError('message exchange authority differs')
    ReportedProviderUsage(CONTRACT, record['thread_id'], usage(record['response']))
    if record['request'] != request_body(config, history, bundle):
        raise ValueError('message request differs from frozen context or grants external tools/state')
    submission = response_submission(record['response'], expected_model=expected_model)
    return record, submission


def reported_usage(raw, *, expected_thread_id=None):
    try:
        record = _json(raw)
        if record.get('kind') != CONTRACT or expected_thread_id is not None and record.get('thread_id') != expected_thread_id:
            return None
        return ReportedProviderUsage(CONTRACT, record['thread_id'], usage(record['response']))
    except (ValueError, TypeError, KeyError, UnicodeError):
        return None


class ResponsesHTTPTransport:
    """Fixed TLS endpoint, no redirects/retries, credentials excluded from retained bodies."""
    def __init__(self, timeout_seconds):
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError('Responses timeout must be positive')
        self.timeout_seconds = timeout_seconds

    def __call__(self, payload):
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('OPENAI_API_KEY is required for a live message provider')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        request = urllib.request.Request(ENDPOINT, data=payload, method='POST',
            headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        opener = urllib.request.build_opener(NoRedirect())
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            # Keep native error/usage when supplied, never an authorization header.
            body = error.read().replace(key.encode(), b'[redacted credential]')
            return body


@dataclass(frozen=True)
class MessageQualification:
    _bytes: bytes

    @classmethod
    def from_dict(cls, document):
        fields = {'schema_version','provider_revision','configuration','scope','exchanges'}
        if (not isinstance(document, Mapping) or set(document) != fields or type(document['schema_version']) is not int or document['schema_version'] != 1
            or document['provider_revision'] != REVISION
            or document['scope'] not in {'zero_gpu_contract_fixture_only','live_two_turn_message_provider'}
            or not isinstance(document['exchanges'], list) or len(document['exchanges']) != 2):
            raise ValueError('message qualification fields or scope differ')
        validate_configuration(document['configuration'])
        history, thread, model, response_ids = [], None, None, set()
        for turn, record in enumerate(document['exchanges'], 1):
            bundle = qualification_bundle(turn)
            observed, submission = validate_exchange(canonical_json_bytes(record), config=document['configuration'],
                history=history, bundle=bundle, run_id='provider-qualification',turn=turn,thread_id=thread,expected_model=model or document['configuration']['model'])
            candidates = _project_candidate_submission(submission,submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                arm='provider-qualification',environment_kind='open_cake',maximum_candidates_per_turn=1)
            if candidates != (canonical_json_bytes({'qualification_turn':turn}),) or observed['response']['id'] in response_ids:
                raise ValueError('message qualification did not preserve two independent turns')
            thread, model = observed['thread_id'], observed['response']['model']
            response_ids.add(observed['response']['id'])
            history = observed['request']['input'] + observed['response']['output']
        return cls(canonical_json_bytes(document))

    @classmethod
    def load(cls, path):
        return cls.from_dict(_json(Path(path).read_bytes()))

    @property
    def document(self): return _json(self._bytes)
    @property
    def canonical_sha256(self): return sha256(self._bytes).hexdigest()
    @property
    def provider_revision(self): return self.document['provider_revision']
    @property
    def configuration_sha256(self): return sha256(canonical_json_bytes(self.document['configuration'])).hexdigest()
    @property
    def scope(self): return self.document['scope']
    @property
    def reported_model(self): return self.document['exchanges'][0]['response']['model']


def qualification_bundle(turn):
    return canonical_json_bytes({'qualification':CONTRACT,'turn':turn,'expected':{
        'schema_version':1,'arm':'provider-qualification','candidates':[{'qualification_turn':turn}]}})


def qualify(config, transport, *, fixture=False, observe=None):
    validate_configuration(config)
    if not fixture and type(transport) is not ResponsesHTTPTransport:
        raise ValueError('live qualification requires the owned HTTP transport')
    history, records, thread = [], [], str(uuid.uuid4())
    for turn in (1,2):
        body = request_body(config,history,qualification_bundle(turn))
        request_bytes = canonical_json_bytes(body)
        if observe is not None: observe(turn, 'request', request_bytes)
        response_bytes = transport(request_bytes)
        if observe is not None: observe(turn, 'response', response_bytes)
        record = {'schema_version':1,'kind':CONTRACT,'run_id':'provider-qualification','turn':turn,
                  'thread_id':thread,'request':body,'response':_json(response_bytes)}
        records.append(record)
        response_submission(record['response'])
        history = body['input'] + record['response']['output']
    return MessageQualification.from_dict({'schema_version':1,'provider_revision':REVISION,
        'configuration':config,'scope':'zero_gpu_contract_fixture_only' if fixture else 'live_two_turn_message_provider',
        'exchanges':records})


class ResponsesRunProvider:
    """Only conversation history is owned here; Run lifecycle and budgets stay in Lab."""
    def __init__(self, *, qualification, task_packages, transport=None):
        self.qualification_sha256 = qualification.canonical_sha256
        self.provider_revision = qualification.provider_revision
        self.configuration = qualification.document['configuration']
        self._model = qualification.reported_model
        if transport is None:
            if qualification.scope != 'live_two_turn_message_provider':
                raise ValueError('fixture qualification cannot authorize a live Responses call')
            transport = ResponsesHTTPTransport(self.configuration['timeout_seconds'])
        if type(transport) is ResponsesHTTPTransport and qualification.scope != 'live_two_turn_message_provider':
            raise ValueError('fixture qualification cannot authorize a live Responses call')
        if type(transport) is ResponsesHTTPTransport and transport.timeout_seconds != self.configuration['timeout_seconds']:
            raise ValueError('message transport timeout differs from its qualification')
        self._transport = transport
        self._packages = dict(task_packages)
        self._records = {name: [] for name in task_packages}

    def turn(self, request):
        package = self._packages.get(request.run_id)
        if (package is None or package.arm != request.arm or package.environment_kind != request.environment_kind
            or request.state_card is None):
            raise ValueError('message provider Run or task binding differs')
        previous = self._records[request.run_id]
        if request.turn != len(previous)+1 or request.cumulative_provider_tokens != sum(usage(r['response']) for r in previous):
            raise ValueError('message provider history or native budget differs')
        thread = previous[0]['thread_id'] if previous else str(uuid.uuid4())
        if request.thread_id != (thread if previous else None):
            raise ValueError('message provider resume identity differs')
        history = previous[-1]['request']['input'] + previous[-1]['response']['output'] if previous else []
        bundle = package.evidence_bundle(request.state_card)
        body = request_body(self.configuration, history, bundle)
        record = {'schema_version':1,'kind':CONTRACT,'run_id':request.run_id,'turn':request.turn,
                  'thread_id':thread,'request':body,'response':None}
        response_bytes = None
        try:
            response_bytes = self._transport(canonical_json_bytes(body))
            record['response'] = _json(response_bytes)
            raw = canonical_json_bytes(record)
            _, submission = validate_exchange(raw,config=self.configuration,history=history,bundle=bundle,
                run_id=request.run_id,turn=request.turn,thread_id=thread,expected_model=self._model)
            if record['response']['id'] in {item['response']['id'] for item in previous}:
                raise ValueError('message provider reused a native response')
            candidates = _project_candidate_submission(submission,submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                arm=request.arm,environment_kind=request.environment_kind,maximum_candidates_per_turn=request.maximum_candidates_per_turn)
        except Exception as error:
            raw = canonical_json_bytes(record)
            raise RunProtocolFault('provider_fault',str(error),artifact_payloads={'provider_stdout':raw,
                **({'provider_response':response_bytes} if isinstance(response_bytes,bytes) else {})},
                reported_usage=reported_usage(raw,expected_thread_id=thread)) from error
        previous.append(record)
        return ProviderTurn(thread,usage(record['response']),candidates,
            tuple(sha256(item).hexdigest() for item in candidates),submission,raw,sha256(raw).hexdigest(),
            submission.decode('utf-8'),1,'single_exact',reference_bundle=bundle)
