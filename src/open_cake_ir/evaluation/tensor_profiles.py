"""Typed registration of in-evaluate tensor profile formats.

Platform modules own their record semantics. The shared receipt reader dispatches
once by the declared kind; it never interprets another platform's raw activity.
"""
from types import MappingProxyType
from .hip_observations import HIP_PROFILE
from .metax_observations import MACA_PROFILE
from .metax_program_profile import MACA_PROGRAM_PROFILE

TENSOR_PROFILES = MappingProxyType({source.kind: source for source in (HIP_PROFILE, MACA_PROFILE, MACA_PROGRAM_PROFILE)})
