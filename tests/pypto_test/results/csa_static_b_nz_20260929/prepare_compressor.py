"""逐张生成Compressor NZ私有候选，每项只改变一张静态B。"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('weight', choices=('cmp_wkv', 'cmp_wgate', 'inner_wkv', 'inner_wgate'))
    name = p.parse_args().weight
    inner = name.startswith('inner_')
    field = name.split('_', 1)[1]
    width = 256 if inner else 1024
    module = 'inner' if inner else 'main'
    kernel_file = 'decode_indexer_compressor.py' if inner else 'decode_compressor_ratio4.py'
    root_shape = 'INNER_OUT_DIM' if inner else 'MAIN_OUT_DIM'
    dest = ROOT.parent / f'csa_{name}_nz_20260929'
    dest.mkdir(exist_ok=True)
    text = (ROOT.parent / 'csa_idx_q_nz_20260929/prepare.py').read_text()
    text = text.replace('Indexer Q的INT8 B', name + '的BF16 B')
    text = text.replace('csa-idx-q-nz', 'csa-' + name.replace('_', '-') + '-nz')
    text = text.replace('dsv4_csa_idx_q_nz', 'dsv4_csa_' + name + '_nz')
    start, end = text.index('def kernel(text):'), text.index('\n\ndef layout(text):')
    body = f'''def kernel(text):
    old = '{field}: pl.Tensor[[OUT_DIM, D], pl.BF16]'
    assert text.count(old) >= 2
    text = text.replace(old, old[:-1] + ', BF16_WEIGHT_LAYOUT]')
    return once(text, 'import pypto.language as pl\\n',
                'import pypto.language as pl\\n\\nfrom .nz_mode import BF16_WEIGHT_LAYOUT\\n')
'''
    text = text[:start] + body + text[end:]
    text = text.replace('"idx_wq_b"', '"' + name + '"')
    start, end = text.index('def adapter(text):'), text.index('\n\ndef main():')
    body = f'''def adapter(text):
    text = once(text, 'def root_weight(name, shape, dtype):', 'def root_weight(name, shape, dtype, module=None):')
    text = once(text, 'value = getattr(attention, name).weight.detach()',
                'value = (getattr(attention, name) if module is None else module).weight.detach()')
    text = once(text, 'return value if current == 29 else _recast(name, value, current, 29)',
                'if current == 29:\\n                return value\\n'
                '            if module is not None:\\n'
                '                logger.info("PTO_CSA_WEIGHT_INIT_NZ %s: PTO专用NZ副本，Native原权重保留", name)\\n'
                '                return torch_npu.npu_format_cast(value, 29)\\n'
                '            return _recast(name, value, current, 29)')
    return once(text, '"{name}": weight({module}.{field}, ({width}, 4096), bf16),',
                '"{name}": root_weight("{name}", ({width}, 4096), bf16, {module}.{field}),')
'''
    text = text[:start] + body + text[end:]
    path = 'layer.self_attn.' + ('indexer.compressor' if inner else 'compressor') + '.' + field + '.weight'
    text = text.replace('layer.self_attn.indexer.wq_b.weight', path)
    text = text.replace("{PACKAGE}/decode_indexer.py", '{PACKAGE}/' + kernel_file)
    text = text.replace('idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8]',
                        f'{name}: pl.Tensor[[{root_shape}, D], pl.BF16]')
    text = text.replace('idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT]',
                        f'{name}: pl.Tensor[[{root_shape}, D], pl.BF16, BF16_WEIGHT_LAYOUT]')
    text = text.replace('Native原地址', '初始化准备PTO专用NZ，Native原权重不变')
    (dest / 'prepare.py').write_text(text)
    print(dest / 'prepare.py')


if __name__ == '__main__':
    main()
