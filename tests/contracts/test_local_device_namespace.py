"""A container ordinal never changes the physical lock or admitted PCI device."""
import ctypes
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation import local_broker, triton_metax


class DeviceNamespaceTests(unittest.TestCase):
    def test_runtime_selection_keeps_physical_lock_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch.object(local_broker, '_lock_path', return_value=Path(directory)/'legacy.lock'):
            job = local_broker.admit_local_job('maca', device=7, lock_scope='device',
                runtime_device=0, expected_pci='0000:d7:00')
            fd = int(os.environ['METAL_BROKER_LOCK_FD'])
            barrier = int(os.environ['OPEN_CAKE_LOCAL_LEGACY_FD'])
            try:
                self.assertEqual(local_broker.observe_local_job('maca'), job)
                self.assertEqual(os.environ['MACA_VISIBLE_DEVICES'], '0')
                self.assertEqual(os.environ['OPEN_CAKE_LOCAL_DEVICE'], '7')
                self.assertEqual(os.fstat(fd).st_ino, (Path(directory)/'legacy-device-7.lock').stat().st_ino)
                # Another runtime namespace still contends on the same physical card.
                with patch.dict(os.environ, {}, clear=True), self.assertRaises(local_broker.LocalBrokerBusy):
                    local_broker.admit_local_job('maca', device=7, lock_scope='device',
                        runtime_device=3, expected_pci='0000:d7:00')
                os.environ['OPEN_CAKE_LOCAL_RUNTIME_DEVICE'] = '1'
                with self.assertRaisesRegex(ValueError, 'mapping differs'):
                    local_broker.observe_local_job('maca')
                os.environ['OPEN_CAKE_LOCAL_RUNTIME_DEVICE'] = '0'
                os.environ.pop('OPEN_CAKE_LOCAL_EXPECTED_PCI')
                with self.assertRaisesRegex(ValueError, 'exact PCI binding'):
                    local_broker.observe_local_job('maca')
            finally:
                os.close(fd)
                os.close(barrier)

    def test_partial_or_invalid_namespace_is_refused_before_lock(self):
        for kind, device, scope, runtime, pci in (
            ('maca', 1, 'device', 0, None), ('maca', 1, 'device', None, '0000:34:00'),
            ('maca', None, 'device', 0, '0000:34:00'), ('maca', 1, 'user', 0, '0000:34:00'),
            ('maca', 1, 'device', True, '0000:34:00'), ('maca', 1, 'device', 0, 'invalid'),
            ('cuda', 1, 'device', 0, '0000:34:00'),
        ):
            with self.subTest(binding=(kind, device, scope, runtime, pci)), \
                 patch.object(local_broker, '_acquire') as acquire:
                with self.assertRaises(ValueError):
                    local_broker.admit_local_job(kind, device=device, lock_scope=scope,
                        runtime_device=runtime, expected_pci=pci)
                acquire.assert_not_called()

    def test_actual_pci_mismatch_refuses_device_admission(self):
        target = triton_metax.declared_target('xcore1002')
        properties = SimpleNamespace(name=target.device_names[0], warp_size=64,
            pci_domain_id=0, pci_bus_id=0x48, pci_device_id=0)
        torch = SimpleNamespace(version=SimpleNamespace(maca='fixture'), cuda=SimpleNamespace(
            device_count=lambda: 1, get_device_properties=lambda index: properties))
        driver = SimpleNamespace(active=SimpleNamespace(get_current_target=lambda:
            SimpleNamespace(backend='maca', arch=target.triton_arch, warp_size=64)))
        def native_query(pointer, attribute, device):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = {75: 10, 76: 2}[attribute]
            return 0
        api = SimpleNamespace(mcDeviceGetAttribute=Mock(side_effect=native_query))
        with patch.dict('sys.modules', {'torch': torch, 'triton.runtime': SimpleNamespace(driver=driver)}), \
             patch.object(local_broker, 'observe_local_job', return_value='maca-123456789abc'), \
             patch.object(triton_metax.ctypes, 'CDLL', return_value=api), \
             patch.dict(os.environ, {'OPEN_CAKE_LOCAL_EXPECTED_PCI': '0000:34:00'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'expected 0000:34:00, observed 0000:48:00'):
                triton_metax.observe_local_metax('xcore1002', runtime_library='/fixture/libmaca.so')
            os.environ['OPEN_CAKE_LOCAL_EXPECTED_PCI'] = '0000:48:00'
            observed = triton_metax.observe_local_metax('xcore1002', runtime_library='/fixture/libmaca.so')
            self.assertEqual(observed.pci_bus_id, '0000:48:00')
