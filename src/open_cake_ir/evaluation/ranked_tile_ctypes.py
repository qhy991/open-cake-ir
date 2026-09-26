"""Load the compiled B300 ranked-tile pointer ABI for Evaluation binding.

The caller proves the shared library was built from `lowered.source`, owns
device tensors and provides an isolated broker-held process. No GPU allocation
or launch occurs until `RankedTileExecutable.bind` is invoked by Evaluation.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

from open_cake_ir.compiler.backends.native_cuda_ranked_tile import (
    NativeRankedTileLowering,
)
from .ranked_tile_launch import RankedTileBound, RankedTileExecutable


def load_ranked_tile_ctypes(lowered: NativeRankedTileLowering,
                            library_path: Path, *, pointer_of,
                            isolated_process: bool) -> RankedTileExecutable:
    """Bind named C functions while keeping tensor ownership with the caller."""
    lowered.validate_binding()
    if not callable(pointer_of) or type(isolated_process) is not bool:
        raise ValueError('ranked tile loader needs pointer and process owners')
    library=ctypes.CDLL(str(library_path.resolve(strict=True)))
    names=lowered.toolchain_requirements['host_abi']
    required={'ranks','source_events','bin_bytes','output_bytes',
              'create','launch','destroy','stolen','payloads'}
    if set(names)!=required:
        raise ValueError('ranked tile compiled ABI names differ')
    functions={name:getattr(library,spelling) for name,spelling in names.items()}
    functions['ranks'].restype=ctypes.c_int
    functions['source_events'].restype=ctypes.c_int
    functions['bin_bytes'].restype=ctypes.c_size_t
    functions['output_bytes'].restype=ctypes.c_size_t
    void_array=ctypes.POINTER(ctypes.c_void_p)
    functions['create'].argtypes=[void_array]*6+[ctypes.POINTER(ctypes.c_void_p)]
    functions['create'].restype=ctypes.c_int
    functions['launch'].argtypes=[ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
    functions['launch'].restype=ctypes.c_int
    functions['destroy'].argtypes=[ctypes.c_void_p]
    functions['destroy'].restype=ctypes.c_int
    for name in ('stolen','payloads'):
        functions[name].argtypes=[ctypes.c_void_p,ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int)]
        functions[name].restype=ctypes.c_int
    world=lowered.analysis.world_size
    output_bytes=lowered.analysis.items_per_rank*lowered.analysis.feature_width*2
    bin_bytes=functions['bin_bytes']()
    if (functions['ranks']()!=world or functions['source_events']()!=20
            or functions['output_bytes']()!=output_bytes
            or not (world-1)*lowered.analysis.items_per_rank*
                   lowered.analysis.feature_width*2 < bin_bytes < 16*1024*1024):
        raise ValueError('ranked tile compiled rank, event or storage facts differ')

    def bind(inputs,outputs,contexts):
        del contexts  # The Evaluation owner captures and rechecks these.
        ordered=('hidden','expert_ids','route_weights','w_up_gate','w_down')
        groups=[]
        for name in ordered:
            pointers=(ctypes.c_void_p*world)()
            for rank in range(world):
                value=pointer_of(inputs[rank][name])
                if type(value) is not int or value<=0:
                    raise ValueError(f'rank {rank} input {name!r} pointer differs')
                pointers[rank]=ctypes.c_void_p(value)
            groups.append(pointers)
        output_pointers=(ctypes.c_void_p*world)()
        for rank in range(world):
            value=pointer_of(outputs[rank])
            if type(value) is not int or value<=0:
                raise ValueError(f'rank {rank} output pointer differs')
            output_pointers[rank]=ctypes.c_void_p(value)
        handle=ctypes.c_void_p()
        status=functions['create'](*groups,output_pointers,ctypes.byref(handle))
        if status!=0 or not handle.value:
            raise ValueError(f'ranked tile create status {status!r}')

        def launch(communication,budgets):
            return functions['launch'](
                handle,(ctypes.c_int*world)(*communication),
                (ctypes.c_int*world)(*budgets))

        def observe(name,rank):
            value=ctypes.c_int(-1)
            status=functions[name](handle,rank,ctypes.byref(value))
            if status!=0:
                raise ValueError(f'ranked tile {name} status {status!r}')
            return value.value

        return RankedTileBound(
            launch=launch,
            stolen=lambda rank:observe('stolen',rank),
            payloads=lambda rank:observe('payloads',rank),
            destroy=lambda:functions['destroy'](handle),
        )

    return RankedTileExecutable(world,20,output_bytes,isolated_process,bind)
