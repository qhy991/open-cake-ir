"""Triton's native-authoring implementation, bound explicitly by its registry row."""


class TritonNativeAdapter:
    def isolated_compiler(self, config):
        from .triton_build import IsolatedTritonCompiler
        return IsolatedTritonCompiler(**config)

    def builder(self, *, workload, case_id, isolated_compiler):
        from .environments import TritonToolchainBuilder
        return TritonToolchainBuilder(workload=workload, case_id=case_id,
                                      isolated_compiler=isolated_compiler)

    def environment(self, builder, **kwargs):
        from .environments import NativeTritonEnvironment
        return NativeTritonEnvironment(builder, **kwargs)

    def source(self, source, requirements):
        from open_cake_ir.compiler.toolchain import project_triton_kernel
        return project_triton_kernel(source, requirements)

    def block(self, requirements, *, warp_size):
        return [requirements["compile_options"]["num_warps"] * warp_size, 1, 1]

    def baseline(self, source, requirements):
        return {"kernel_source": source, "compile_constants": dict(requirements["compile_constants"]),
                "compile_options": dict(requirements["compile_options"]),
                "grid": list(requirements["grid"])}
