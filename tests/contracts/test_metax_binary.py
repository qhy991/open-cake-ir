"""Offline MACA bundle refusal cases; fixtures establish no hardware capability."""

import struct
import unittest

from open_cake_ir.compiler.metax_toolchain import device_image, native_pointer_parameters


def native_fixture(pointers=1, name="kernel", *, arguments=None):
    """A metadata-only ELF fixture; its text is not executable GPU code."""
    def pack(value):
        if type(value) is int:
            return bytes([value]) if value < 128 else b"\xcc" + bytes([value])
        if isinstance(value, str):
            payload = value.encode()
            return (bytes([0xA0 + len(payload)]) if len(payload) < 32 else b"\xd9" + bytes([len(payload)])) + payload
        if isinstance(value, list):
            prefix = bytes([0x90 + len(value)]) if len(value) < 16 else b"\xdc" + struct.pack(">H", len(value))
            return prefix + b"".join(pack(item) for item in value)
        if isinstance(value, dict):
            return bytes([0x80 + len(value)]) + b"".join(pack(k) + pack(v) for k, v in value.items())
        raise ValueError("unsupported fixture value")
    args = arguments if arguments is not None else [
        {".arg_param_pass": "global_buffer", ".arg_offset_bytes": 8 * i, ".arg_size_bytes": 8}
        for i in range(pointers)]
    metadata = pack({"macahca.kernels": [{".name": name, ".args": args}]})
    owner = b"MetaX\0"
    note = struct.pack("<III", len(owner), len(metadata), 0x30)
    note += owner + bytes((-len(owner)) % 4) + metadata + bytes((-len(metadata)) % 4)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", header, 18, 253)
    directory_offset = 64 + len(note)
    struct.pack_into("<Q", header, 40, directory_offset)
    struct.pack_into("<HH", header, 58, 64, 1)
    section = bytearray(64)
    struct.pack_into("<I", section, 4, 7)
    struct.pack_into("<QQ", section, 24, 64, len(note))
    return bytes(header) + note + bytes(section)


def bundle(*, architecture="xcore1000", native=None, extra=None):
    if native is None:
        native = bytearray(64)
        native[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", native, 18, 253)
        native = bytes(native)
    base = "maca-mxc-metax-macahca--" + architecture
    rows = [("host-x86_64-unknown-linux-gnu", b""),
            (base + "-bc", b"synthetic-bitcode"), (base, native)]
    if extra is not None:
        rows.append(extra)
    header = b"__CLANG_OFFLOAD_BUNDLE__" + struct.pack("<Q", len(rows))
    offset = len(header) + sum(24 + len(name) for name, _ in rows)
    body = b""
    for name, payload in rows:
        header += struct.pack("<QQQ", offset, len(payload), len(name)) + name.encode()
        body += payload
        offset += len(payload)
    return header + body, native


class MetaxBinaryTests(unittest.TestCase):
    def test_paired_baseline_admission_reads_declared_native_family_and_kernel(self):
        from hashlib import sha256
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from open_cake_ir.compiler.backends.triton import target_route_facts
        from open_cake_ir.compiler.target import Target
        from open_cake_ir.evaluation.core import LaunchableCandidate, TensorLaunchManifest
        from open_cake_ir.evaluation.paired import candidate_identity
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.lab.admission import validate_paired_baseline
        from open_cake_ir.serialization import canonical_json_bytes as encoded
        from open_cake_ir.tasks.workloads import create_task

        root = Path(__file__).resolve().parents[2]
        target = Target.load(root / 'compiler/targets/xcore1002.json')
        document, _ = create_task('rmsnorm', backend='triton-metax', rows=2, columns=128)
        workload = WorkloadContract(document)
        source = b'import triton\nimport triton.language as tl\n@triton.jit\ndef kernel(x, gamma, y):\n    offsets = tl.arange(0, 128)\n    tl.store(y + offsets, tl.load(x + offsets))\n'
        requirements = {'compiler': 'triton', 'target': target.target_id,
            'kernel_entry_point': 'kernel', 'grid': [2, 1, 1], 'compile_options': {'num_warps': 1},
            'signature': {name: '*fp32' for name in ('x', 'gamma', 'y')}, 'compile_constants': {},
            **target_route_facts(target)}
        lowering = SimpleNamespace(source=source.decode(), toolchain_requirements=requirements)

        def check(pointer_count, hidden, native_name='kernel'):
            manifest = TensorLaunchManifest.for_workload(workload, 'primary', target=target.target_id,
                kernel_name='kernel', grid=requirements['grid'], block=[target.warp_size, 1, 1],
                dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=hidden)
            payloads = {'lowered_source': source, 'launch_manifest': encoded(manifest.as_dict()),
                'ttgir': b'tt.func public @kernel(%x: !tt.ptr<f32>, %g: !tt.ptr<f32>, %y: !tt.ptr<f32>) attributes {}',
                'mcfatbin': bundle(native=native_fixture(pointer_count, native_name))[0]}
            candidate = LaunchableCandidate('a' * 64, target.target_id, 'kernel',
                {key: sha256(value).hexdigest() for key, value in payloads.items()},
                manifest.canonical_sha256, payloads)
            with patch('open_cake_ir.lab.admission.load_baseline_bundle', return_value=candidate):
                validate_paired_baseline(project_root=root, workload=workload,
                    evaluation={'case_id': 'primary'}, route={'backend': 'triton'},
                    execution={'fixed_baseline': {'bundle_path': 'fixture', 'candidate': candidate_identity(candidate)}},
                    baseline_lowering=lowering, manifest_parser=TensorLaunchManifest.from_dict)

        # Actual admission calls the native parser, not a mock of the inspected fields.
        for pointers, hidden in ((3, 0), (5, 2)):
            with self.subTest(pointers=pointers):
                check(pointers, hidden)
        with self.assertRaisesRegex(ValueError, 'hidden pointer commitments'):
            check(5, 0)
        with self.assertRaisesRegex(ValueError, 'kernel name'):
            check(5, 2, 'another')

    def test_sealing_derives_scratch_from_native_metadata(self):
        from types import SimpleNamespace
        from open_cake_ir.lab.build import _hidden_pointers
        route = SimpleNamespace(gpu_backend="maca", binary_role="mcfatbin", text_role="ttgir")
        stages = {"ttgir": b"tt.func public @kernel(%x: !tt.ptr<f32>) attributes {}"}
        for declared, expected in ((1, 0), (3, 2)):
            stages["mcfatbin"] = bundle(native=native_fixture(declared))[0]
            self.assertEqual(_hidden_pointers(route, stages, 1, codegen_arch="xcore1000", kernel_name="kernel"), expected)
        stages["mcfatbin"] = bundle(native=native_fixture(2))[0]
        with self.assertRaisesRegex(ValueError, "scratch pointer count"):
            _hidden_pointers(route, stages, 1, codegen_arch="xcore1000")

    def test_native_metadata_counts_launcher_slots_and_checks_kernel(self):
        for count in (1, 3, 5):
            payload = bundle(native=native_fixture(count))[0]
            self.assertEqual(native_pointer_parameters(payload, "xcore1000", "kernel"), count)
            with self.assertRaisesRegex(ValueError, "kernel name"):
                native_pointer_parameters(payload, "xcore1000", "another")

    def test_runtime_hidden_fields_are_not_launcher_parameters(self):
        args = [{".arg_param_pass": "global_buffer", ".arg_offset_bytes": 0, ".arg_size_bytes": 8},
                {".arg_param_pass": "hidden_global_offset_x", ".arg_offset_bytes": 8, ".arg_size_bytes": 8}]
        self.assertEqual(native_pointer_parameters(bundle(native=native_fixture(arguments=args))[0], "xcore1000"), 1)
        args.reverse()
        with self.assertRaises(ValueError):
            native_pointer_parameters(bundle(native=native_fixture(arguments=args))[0], "xcore1000")

    def test_missing_note_scalar_and_bad_pointer_offsets_are_refused(self):
        with self.assertRaisesRegex(ValueError, "section directory"):
            native_pointer_parameters(bundle()[0], "xcore1000")
        for kind, size, offset in (("by_value", 8, 0), ("global_buffer", 4, 0), ("global_buffer", 8, 8)):
            args = [{".arg_param_pass": kind, ".arg_size_bytes": size, ".arg_offset_bytes": offset}]
            with self.assertRaises(ValueError):
                native_pointer_parameters(bundle(native=native_fixture(arguments=args))[0], "xcore1000")

    def test_returns_only_the_native_member_of_the_requested_family(self):
        payload, native = bundle()
        self.assertEqual(device_image(payload, "xcore1000"), native)
        with self.assertRaisesRegex(ValueError, "does not declare only"):
            device_image(payload, "xcore1002")

    def test_every_truncated_directory_or_native_image_is_refused(self):
        payload, _ = bundle()
        for end in (0, 23, 24, 31, 32, 55, 100, len(payload) - 1):
            with self.subTest(end=end), self.assertRaises(ValueError):
                device_image(payload[:end], "xcore1000")

    def test_a_device_elf_cannot_be_relabelled_as_mxc(self):
        wrong = bytearray(bundle()[1])
        struct.pack_into("<H", wrong, 18, 62)  # x86-64
        for native in (bytes(wrong), b"\x7fELF\x02\x01", b""):
            with self.subTest(native=native), self.assertRaises(ValueError):
                device_image(bundle(native=native)[0], "xcore1000")

    def test_ambiguous_or_additional_architectures_are_refused(self):
        base = "maca-mxc-metax-macahca--"
        for extra in ((base + "xcore1002", bundle()[1]),
                      (base + "xcore1000", bundle()[1])):
            with self.subTest(extra=extra[0]), self.assertRaises(ValueError):
                device_image(bundle(extra=extra)[0], "xcore1000")

    def test_payload_must_not_point_into_the_directory_or_overlap(self):
        payload, _ = bundle()
        # Locate the two device records from the documented binary directory, then
        # corrupt their offsets without changing the payload or architecture names.
        first = 24 + 8
        second = first + 24 + len("host-x86_64-unknown-linux-gnu")
        third = second + 24 + len("maca-mxc-metax-macahca--xcore1000-bc")
        for offset in (0, struct.unpack_from("<Q", payload, second)[0]):
            corrupted = bytearray(payload)
            struct.pack_into("<Q", corrupted, third, offset)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                device_image(bytes(corrupted), "xcore1000")


if __name__ == "__main__":
    unittest.main()
