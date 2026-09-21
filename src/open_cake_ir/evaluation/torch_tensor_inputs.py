"""Native tensor input storage for sealed kernels and ordered Programs.

No task mathematics or oracle lives here. CPU tensors cross the device boundary as
bytes, preserving FP8 storage and avoiding expansion into Python float objects.
"""
from types import MappingProxyType

from .core import _MODULE_LOADERS, _TORCH_DTYPE_NAMES, load_torch_program
from .platforms import platform_for


def check_cpu_tensor_inputs(manifest, inputs):
    import torch
    expected = {name: (shape, dtype) for name, shape, dtype, mode in manifest.tensor_abi if mode == 'input'}
    if set(inputs) != set(expected):
        raise ValueError('native tensor input names differ from the sealed ABI')
    for name, (shape, dtype) in expected.items():
        value = inputs[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or tuple(value.shape) != shape or not value.is_contiguous()
                or value.dtype != getattr(torch, _TORCH_DTYPE_NAMES[dtype])):
            raise ValueError(f'native CPU input {name!r} shape, dtype, device or storage differs')


class LoadedTorchTensorInputs:
    """One CPU tensor case bound to the existing native module/Program owners."""

    def __init__(self, candidate, manifest, inputs, admission):
        import torch
        if (candidate.target != manifest.target or candidate.entry_point != manifest.kernel_name
                or candidate.launch_spec_sha256 != manifest.canonical_sha256):
            raise ValueError('native tensor candidate differs from its sealed manifest')
        check_cpu_tensor_inputs(manifest, inputs)
        loader = _MODULE_LOADERS.get(platform_for(candidate.target).code_object)
        if loader is None:
            raise ValueError('the target has no native tensor module loader')
        self.candidate, self.manifest, self.admission = candidate, manifest, admission
        self.inputs = MappingProxyType(dict(inputs))
        self.arguments = []
        for name, shape, dtype, mode in manifest.tensor_abi:
            torch_dtype = getattr(torch, _TORCH_DTYPE_NAMES[dtype])
            if mode == 'input':
                value = inputs[name].view(torch.uint8).to('cuda:0').view(torch_dtype).reshape(shape)
            else:
                # These tensor-oracle tasks have bounded outputs or IEEE NaN/Inf.
                # A finite poison cannot masquerade as their required NaN/Inf.
                poison = -(2**31) if dtype == 'int32' else torch.finfo(torch_dtype).max
                value = torch.full(shape, poison, dtype=torch_dtype, device='cuda:0')
            self.arguments.append(value)
        for (name, _, _, mode), value in zip(manifest.tensor_abi, self.arguments, strict=True):
            if mode == 'input' and not torch.equal(value.view(torch.uint8).cpu(), inputs[name].view(torch.uint8)):
                raise ValueError(f'native input {name!r} bytes differ before launch')
        if candidate.is_program:
            self.loaded, self.buffers = load_torch_program(candidate, manifest, self.arguments, admission, loader)
        elif manifest.aligned_variant:
            from .kernel_bundle import LoadedAlignmentCandidate
            self.loaded = LoadedAlignmentCandidate(candidate, manifest, admission, loader, torch.cuda.synchronize)
            self.buffers = MappingProxyType(dict(zip((row[0] for row in manifest.tensor_abi), self.arguments)))
        else:
            self.loaded = loader(candidate, manifest, admission)
            self.buffers = MappingProxyType(dict(zip((row[0] for row in manifest.tensor_abi), self.arguments)))

    def launch(self):
        import torch
        self.loaded.launch(self.arguments, tensor_contract=self.manifest,
                           stream=torch.cuda.current_stream().cuda_stream)

    def snapshot(self):
        """Return complete CPU outputs and exact public input byte-effect checks."""
        import torch
        outputs, checks = {}, {}
        for (name, shape, dtype, mode), value in zip(self.manifest.tensor_abi, self.arguments, strict=True):
            raw = value.detach().view(torch.uint8).cpu()
            if mode == 'input':
                checks[name] = bool(torch.equal(raw, self.inputs[name].view(torch.uint8)))
            else:
                outputs[name] = raw.view(value.dtype).reshape(shape)
        return outputs, checks

    def close(self):
        import torch
        self.loaded.close(synchronize=torch.cuda.synchronize)
