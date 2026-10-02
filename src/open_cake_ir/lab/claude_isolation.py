"""Publish a self-contained Claude launcher with an OS enforced workspace boundary.

The published executable includes its configuration; provider qualification pins
that executable using the existing provider identity boundary. No project source,
evidence roots, GPU nodes, ambient home or other Run is mounted into the author.
"""
from pathlib import Path
import json

CLAUDE_WORKSPACE_V1 = 'linux_claude_workspace_v1'

# A standalone launcher so the jailed author needs no imports from this repository.
LAUNCHER = r'''
import json, os, pathlib, subprocess, sys

POLICY = 'linux_claude_workspace_v1'

def command(workspace, program):
    args = [CONFIG['bubblewrap'], '--die-with-parent', '--unshare-all', '--share-net',
            '--cap-drop', 'ALL', '--new-session', '--clearenv']
    for path in ('/usr', '/bin', '/lib', '/lib64'):
        if pathlib.Path(path).exists(): args += ['--ro-bind', path, path]
    args += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
             '--ro-bind', CONFIG['native_executable'], '/provider/claude',
             '--bind', str(workspace), str(workspace), '--chdir', str(workspace)]
    for path in ('/etc/resolv.conf', '/etc/hosts', '/etc/ssl/certs'):
        if pathlib.Path(path).exists(): args += ['--ro-bind', path, path]
    for name,value in {'HOME':str(workspace/'.claude-home'), 'PATH':'/usr/bin:/bin',
        'LANG':'C.UTF-8', 'CUDA_VISIBLE_DEVICES':'-1', 'MACA_VISIBLE_DEVICES':'-1',
        'HIP_VISIBLE_DEVICES':'-1'}.items(): args += ['--setenv', name, value]
    # Authentication is inherited from the trusted launcher environment, never
    # placed in argv, logs, a workspace file, or the probe result.
    for name in ('ANTHROPIC_AUTH_TOKEN','ANTHROPIC_BASE_URL','ANTHROPIC_MODEL',
                 'ANTHROPIC_DEFAULT_OPUS_MODEL','ANTHROPIC_DEFAULT_SONNET_MODEL',
                 'ANTHROPIC_DEFAULT_HAIKU_MODEL'):
        args += ['--setenv', name, os.environ[name]] if name != 'ANTHROPIC_AUTH_TOKEN' else []
    # bwrap --clearenv cannot inherit a single secret without argv. Pass it over
    # stdin to the trusted inner Python shim instead; Claude receives DEVNULL.
    return args + ['--', '/usr/bin/python3', '-c',
        'import os,sys; os.environ["ANTHROPIC_AUTH_TOKEN"]=sys.stdin.readline().rstrip("\\n"); '
        'fd=os.open("/dev/null",os.O_RDONLY); os.dup2(fd,0); os.close(fd); os.execv(sys.argv[1],sys.argv[1:])',
        *program]

def main():
    if sys.argv[1:] in (['--help'], ['--version']):
        os.execv(CONFIG['native_executable'],[CONFIG['native_executable'],*sys.argv[1:]])
    workspace = pathlib.Path.cwd().resolve(strict=True)
    if not workspace.is_dir() or workspace == pathlib.Path('/'):
        raise ValueError('isolated author requires its independent workspace')
    token = pathlib.Path(CONFIG['credential_source']).read_text().strip()
    if not token or '\n' in token: raise ValueError('private provider credential is malformed')
    os.environ.update(ANTHROPIC_AUTH_TOKEN=token, ANTHROPIC_BASE_URL=CONFIG['endpoint'],
        ANTHROPIC_MODEL=CONFIG['model'], ANTHROPIC_DEFAULT_OPUS_MODEL=CONFIG['model'],
        ANTHROPIC_DEFAULT_SONNET_MODEL=CONFIG['model'], ANTHROPIC_DEFAULT_HAIKU_MODEL=CONFIG['model'])
    home = workspace/'.claude-home'
    if home.is_symlink(): raise ValueError('author home must not be a symlink')
    home.mkdir(mode=0o700,exist_ok=True)
    if sys.argv[1:] == ['--cake-isolation-probe']:
        probe = r"""import json, os, pathlib
workspace=pathlib.Path.cwd()
checks={}
for name,path in {'ambient_home':'/root/.claude', 'project':'/root/open-cake-ir',
    'experiments':'/root/open-cake-experiments/sources', 'host_root':'/proc/1/root/root',
    'gpu_nodes':'/dev/mxcd', 'escape':str(workspace/'.isolation-escape')}.items():
    try: os.stat(path); checks[name]=False
    except (FileNotFoundError,PermissionError): checks[name]=True
checks['home_is_local']=pathlib.Path.home()==workspace/'.claude-home'
checks['gpu_device_nodes_absent']=not any(p.name.startswith(('dri','mx','nvidia','kfd')) for p in pathlib.Path('/dev').iterdir())
checks['workspace_writable']=os.access(workspace,os.W_OK)
print(json.dumps({'policy':'linux_claude_workspace_v1','checks':checks}))
raise SystemExit(0 if all(checks.values()) else 1)
"""
        escape=workspace/'.isolation-escape'
        if escape.exists() or escape.is_symlink(): raise ValueError('probe path already exists')
        escape.symlink_to(CONFIG['credential_source'])
        try:
            return subprocess.run(command(workspace,['/usr/bin/python3','-c',probe]),
                input=(token+'\n').encode(),env=os.environ).returncode
        finally: escape.unlink()
    return subprocess.run(command(workspace,['/provider/claude',*sys.argv[1:]]),
        input=(token+'\n').encode(),env=os.environ).returncode

if __name__=='__main__': raise SystemExit(main())
'''


def publish_launcher(path, configuration):
    fields = {'bubblewrap','native_executable','credential_source','endpoint','model'}
    if set(configuration) != fields or any(not isinstance(v,str) or not v for v in configuration.values()):
        raise ValueError('isolated Claude launcher configuration differs')
    for name in ('bubblewrap','native_executable','credential_source'):
        source = Path(configuration[name])
        if not source.is_absolute() or not source.is_file() or source.is_symlink():
            raise ValueError(f'isolated Claude {name} must be a canonical regular file')
    path = Path(path)
    with path.open('x') as stream:
        stream.write('#!/usr/bin/python3\nCONFIG = '+repr(dict(configuration))+'\n'+LAUNCHER)
    path.chmod(0o700)
    return path


def probe_launcher(executable, workspace):
    import subprocess
    result = subprocess.run([str(executable),'--cake-isolation-probe'],cwd=workspace,
                            capture_output=True,timeout=30)
    observation = json.loads(result.stdout)
    if result.returncode:
        raise ValueError('Claude OS workspace isolation probe failed: '+result.stderr.decode(errors='replace')[:512])
    validate_probe_observation(observation)
    return observation


def validate_probe_observation(observation):
    required = {'ambient_home','project','experiments','host_root','gpu_nodes','escape',
                'home_is_local','gpu_device_nodes_absent','workspace_writable'}
    if (set(observation) != {'policy','checks'}
        or observation['policy'] != CLAUDE_WORKSPACE_V1 or set(observation['checks']) != required
        or any(v is not True for v in observation['checks'].values())):
        raise ValueError('Claude OS workspace isolation probe failed')
    return observation
