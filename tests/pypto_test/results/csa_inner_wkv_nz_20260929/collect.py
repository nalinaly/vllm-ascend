"""逐项NZ对照：性能后核对八类状态与实际权重存储。"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT.parent / 'csa_sparse_first_pv_20260929/collect.py'
    spec = importlib.util.spec_from_file_location('single_weight_nz_collection', path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    common.ROOT = ROOT
    common.TITLE = '静态B逐项NZ：inner_wkv'
    common.TARGETS = {'indexer_compressor': 'kv_score_proj_0'}
    common.DESCRIPTION = (
        '只改变一张静态矩阵的初始化布局及读取，累计K顺序/任务数量/依赖保持。'
        '完整CSA绝对时间和P95决定是否接入；性能后八类完整状态逐元素检查失败则不采用。'
    )
    common.main()
    bindings = []
    for history, batch in json.loads((ROOT / 'source.json').read_text())['cases']:
        report_path = ROOT / f'h{history}_b{batch}/swimlane/candidate/report.json'
        binding = json.loads(report_path.read_text())['weight_storage_binding']['inner_wkv']
        if not (binding['pto_format'] == 29 and binding['root_layout'] == 'NZ'
                and binding['shape'] == [256, 4096]):
            raise ValueError(f'权重NZ准备未生效: {binding}')
        if False:
            if not (binding['native_format'] == 29 and binding['same_data_ptr']):
                raise ValueError(f'未复用Native已有NZ: {binding}')
        elif binding['native_format'] not in (0, 2) or binding['same_data_ptr']:
            raise ValueError(f'未保持Native ND/PTO初始化NZ分离: {binding}')
        bindings.append({'history': history, 'batch': batch, 'source': str(report_path), 'binding': binding})
    (ROOT / 'bindings.json').write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
