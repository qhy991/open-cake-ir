"""CuTe's native-authoring implementation, bound explicitly by its registry row."""


class CuTeNativeAdapter:
    def isolated_compiler(self, config):
        from .cute_build import IsolatedCuTeCompiler
        return IsolatedCuTeCompiler(**config)

    def builder(self, *, workload, case_id, isolated_compiler):
        from .cute_build import CuTeToolchainBuilder
        return CuTeToolchainBuilder(workload=workload, case_id=case_id,
                                    isolated_compiler=isolated_compiler)

    def environment(self, builder, **kwargs):
        from .environments import NativeCuTeEnvironment
        return NativeCuTeEnvironment(builder, **kwargs)

    def source(self, source, requirements):
        from open_cake_ir.compiler.cute_toolchain import validate_cute_kernel
        validate_cute_kernel(source, requirements)
        return source

    def block(self, requirements, *, warp_size):
        return list(requirements["block"])

    def baseline(self, source, requirements):
        return {"kernel_source": source, "grid": list(requirements["grid"]),
                "block": list(requirements["block"]),
                "dynamic_shared_memory_bytes": requirements["dynamic_shared_memory_bytes"]}
