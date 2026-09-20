"""Typed registration of in-evaluate tensor profile formats.

Platform modules own their record semantics. The shared receipt reader dispatches
once by the declared kind; it never interprets another platform's raw activity.
"""
from types import MappingProxyType
from .hip_observations import HIP_PROFILE

TENSOR_PROFILES = MappingProxyType({source.kind: source for source in (HIP_PROFILE,)})
