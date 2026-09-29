"""复用Sparse双核口径；性能后检查完整状态，并保留四窗与P95。"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        'sparse_rope_collect', ROOT.parent / 'csa_sparse_first_pv_20260929/collect.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.TITLE = 'Sparse逆RoPE整块Gather'
    module.DESCRIPTION = ('仅将最终逆RoPE的16×64 Gather展平1×1024；保留绝对head偏移和全部浮点算术。'
                          'Sparse AIC作为未改控制，核时包含内核互等；不把核内/正式CSA差额当调度开销。')
    module.main()


if __name__ == '__main__':
    main()
