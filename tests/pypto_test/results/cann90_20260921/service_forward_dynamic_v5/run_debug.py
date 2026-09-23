import runpy, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests/pypto_test'))
from dsv4_csa_env import activate
activate()
from pypto.runtime import configure_log
configure_log('debug')
runpy.run_path('tests/pypto_test/dsv4_csa_service_forward.py',run_name='__main__')
