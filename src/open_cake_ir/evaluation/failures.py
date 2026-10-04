"""Versioned failed-worker observations, separate from acceptance receipts."""
from collections.abc import Mapping
from pathlib import PurePosixPath
import re

WORKER_RESULT_FIELDS = frozenset({'schema_version','job_id','mode','admitted','error','failure_class','counters','receipt'})


def failure_artifacts(result):
    """Return declared failure paths after validating the closed worker envelope.

    Version 1 retains the existing wire form. Version 2 adds diagnostic artifact
    paths only to failed attempts, never to an accepted Evaluation receipt.
    """
    if not isinstance(result,Mapping) or type(result.get('schema_version')) is not int:
        raise ValueError('evaluator command result fields differ')
    version=result['schema_version']
    if version==1 and set(result)==WORKER_RESULT_FIELDS:
        return {}
    if version!=2 or set(result)!=WORKER_RESULT_FIELDS | {'failure_artifacts'}:
        raise ValueError('evaluator command result fields differ')
    paths=result['failure_artifacts']
    if (result['error'] is None or not isinstance(result['failure_class'],str)
        or not result['failure_class'] or result['receipt'] is not None
        or not isinstance(paths,Mapping) or not paths):
        raise ValueError('failure artifacts require a failed attempt without a receipt')
    for role,path in paths.items():
        if (not isinstance(role,str) or re.fullmatch(r'[a-z][a-z0-9_]*',role) is None
            or not isinstance(path,str) or not path or '\\' in path
            or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts):
            raise ValueError('failure artifact role or path differs')
    if len(set(paths.values()))!=len(paths):
        raise ValueError('failure artifact paths must be distinct')
    return dict(paths)
