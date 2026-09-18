"""What every module loader shares before a vendor's driver is asked for anything.

`cuda_driver.py` and `hip_driver.py` each retain one admitted module for the whole of a
tensor-tile evaluation. The part that is the vendor's -- attributes, launch, teardown --
stays in each driver. The part that is not was written twice, byte for byte after a
rename: the exception that keeps a primary failure beside every teardown failure, and the
check that a sealed candidate, its executable bytes and its launch manifest are one
authority before a module is loaded. A duplicate is a defect while it still agrees.
"""

from __future__ import annotations

from hashlib import sha256


class LifecycleError(RuntimeError):
    """Preserve the primary failure and every failure encountered during teardown."""

    def __init__(
        self, primary: BaseException, teardown: BaseException, *remaining: BaseException
    ) -> None:
        self.primary = primary
        self.teardown = teardown
        self.teardown_errors = (teardown, *remaining)
        super().__init__(
            f"module lifecycle failed in {type(primary).__name__}: {primary}; teardown: "
            + "; ".join(f"{type(error).__name__}: {error}" for error in self.teardown_errors)
        )


def is_elf(payload: object) -> bool:
    """Whether these bytes carry the ELF magic every cubin and hsaco starts with."""
    return isinstance(payload, bytes) and payload.startswith(b"\x7fELF")


def check_launch_authority(candidate, payload: bytes, role: str, manifest) -> None:
    """Refuse a load whose candidate, executable and manifest are not one authority.

    `role` is the executable role the candidate's Target declares; the payload must be
    the ELF the candidate sealed under that role, and the manifest must be the one the
    candidate's launch seal names.
    """
    if (
        not is_elf(payload)
        or candidate.target != manifest.target
        or candidate.entry_point != manifest.kernel_name
        or candidate.launch_spec_sha256 != manifest.canonical_sha256
        or candidate.artifact_roles.get(role) != sha256(payload).hexdigest()
    ):
        raise ValueError("persistent candidate launch authority differs")
