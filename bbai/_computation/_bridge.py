import subprocess
import sys
import pathlib

srcdir = pathlib.Path(__file__).parent.absolute()
exe = str(srcdir / "../bbai")
cmd = [
    exe,
]
handle = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr.fileno(), text=True)
num_lines = int(handle.stdout.readline().strip())
code = ''
for _ in range(num_lines):
    code += handle.stdout.readline()
exec(code)
