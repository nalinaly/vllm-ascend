"""Apply the local pto_eager editable-install repair in spawned test workers."""

import os

if os.environ.get("PTO_EAGER_ROOT"):
    from pto_eager_env import activate_pto_eager

    activate_pto_eager()
