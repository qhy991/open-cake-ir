"""CPU return-value substitute for Compiler emissions; not a native-author validator."""
from open_cake_ir.compiler.toolchain import TritonCompilation, triton_route


class CompilerEmissionFixture:
    def __init__(self): self.requests=[]

    def __call__(self,request):
        if request.source_role!='lowered_source':
            raise AssertionError('fixture requires Compiler emission')
        self.requests.append(request)
        requirements = request.toolchain_requirements
        route = triton_route(requirements)
        artifacts = {role:b'CPU Compiler fixture; not device code' for role in route.artifact_roles}
        artifacts['source'] = b'# CPU synthetic compiler expansion\n'+request.source
        artifacts[route.binary_role] = b'\x7fELF CPU fixture; not launchable'
        return TritonCompilation(request.source,request.target,requirements['kernel_entry_point'],artifacts,
            requirements['compile_options']['num_warps']*requirements['warp_size'],0,'CPU fixture',route.code_object.value)
