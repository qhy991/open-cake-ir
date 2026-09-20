"""Frozen optimization material and transform grants, never a second evidence store."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from open_cake_ir.compiler.program_passes import TRANSFORMATIONS
from open_cake_ir.serialization import canonical_json_bytes


@dataclass(frozen=True)
class OptimizationKnowledge:
    _bytes: bytes

    @classmethod
    def from_dict(cls, document):
        fields = {'schema_version', 'knowledge_id', 'version', 'mechanism', 'references', 'transformations'}
        if (not isinstance(document, Mapping) or set(document) != fields
            or type(document['schema_version']) is not int or document['schema_version'] != 1):
            raise ValueError('optimization knowledge fields differ')
        for field in ('knowledge_id', 'version', 'mechanism'):
            if not isinstance(document[field], str) or not document[field].strip():
                raise ValueError(f'optimization knowledge {field} is required')
        references = document['references']
        if not isinstance(references, list) or not references:
            raise ValueError('optimization knowledge must cite its source or counterexample')
        for reference in references:
            if (not isinstance(reference, Mapping) or set(reference) != {'role', 'locator'}
                or reference['role'] not in {'source', 'counterexample', 'target_observation'}
                or not isinstance(reference['locator'], str) or not reference['locator'].strip()):
                raise ValueError('optimization knowledge evidence reference differs')
        _transform_names(document['transformations'])
        return cls(canonical_json_bytes(document))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_bytes()))

    @property
    def document(self):
        return json.loads(self._bytes)


def _transform_names(names):
    known = {transform.name for transform in TRANSFORMATIONS}
    if (not isinstance(names, list) or any(not isinstance(name, str) or name not in known for name in names)
        or len(names) != len(set(names))):
        raise ValueError('transform references must name distinct declared Compiler transformations')


def validate_knowledge_access(value, *, environment_kind):
    """Materials and callable transforms are orthogonal selections, not E/P modes."""
    if not isinstance(value, Mapping) or set(value) != {'materials', 'transformations'}:
        raise ValueError('Run knowledge access must declare materials and transformations')
    _transform_names(value['transformations'])
    if environment_kind != 'open_cake' and value['transformations']:
        raise ValueError('Compiler transformations require Cake authoring')
    materials = value['materials']
    if not isinstance(materials, list):
        raise ValueError('Run knowledge materials must be a frozen list')
    identities = []
    for material in materials:
        record = OptimizationKnowledge.from_dict(material).document
        identities.append(record['knowledge_id'])
    if len(identities) != len(set(identities)):
        raise ValueError('a Run selects exactly one version of each knowledge unit')


def transformation_surface(granted):
    """The callable API is projected from its Compiler owner, without exposing code."""
    _transform_names(granted)
    return [{'name': item.name, 'parameters': list(item.parameters), 'description': item.description}
            for item in TRANSFORMATIONS if item.name in granted]
