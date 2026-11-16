"""Explicitly scoped user-authorized overhead exception; original FAIL remains intact."""
import json
import subprocess
from pathlib import Path

import torch

from causal_trainability import common_batch_v2 as engine
from causal_trainability.sprint03_live import atomic_json, digest
from causal_trainability.telemetry import ROOT, code_revision

NOTE=ROOT/'notes/sprint05_overhead_exception.md'
OUT=ROOT/'runs/causal-trainability/sprint05_common_exception_20261116'


def main():
    torch.set_num_threads(1)
    revision=code_revision()
    assert not revision['dirty_causal_trainability']
    status=subprocess.run(['git','status','--porcelain','--','scripts/run_sprint05_exception.py','notes/sprint05_overhead_exception.md'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert not status.stdout.strip(),'Freeze runner and exception before execution'
    gate=json.loads((engine.CAL/'summary.json').read_text())
    assert gate['identity']==engine.identity(),'Calibration must match unchanged scientific implementation'
    assert gate['passed'] is False
    assert set(gate['cells'])==set(engine.CELLS)
    for name,cell in gate['cells'].items():
        assert len(cell['pairs'])==3 and all(p['bit_equal'] for p in cell['pairs'])
        assert cell['mean_overhead']<=.40 and cell['median_overhead']<=.40
        for i in range(3):
            rows=[engine.verified(engine.CAL/name/f'pair{i}_{flag}'/'result.json') for flag in (True,False)]
            assert rows[0]['final_sha256']==rows[1]['final_sha256']
    engine.SCI=OUT
    OUT.mkdir(parents=True,exist_ok=True)
    current=dict(identity=engine.identity(),runner_sha256=digest(Path(__file__)),exception_sha256=digest(NOTE),
        original_gate_sha256=digest(engine.CAL/'summary.json'),original_gate_passed=False,
        overhead_exception=True,histories=list(range(6)),repeats=list(range(3)),cells=list(engine.CELLS),
        revision=revision)
    manifest=OUT/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==current
    else:atomic_json(manifest,current)
    for history in range(6):
        for repeat in range(3):
            if (OUT/'STOP').exists():return
            folder=OUT/f'breakout_cnn_s{history}'/f'repeat{repeat}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():continue
            attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}'
            cfg,batch,rng,reference,bodies=engine.group(f'breakout_cnn_s{history}',repeat,'s04-common',attempt)
            rows={}
            for name in engine.CELLS:
                body,mode=name.split('_',1)
                rows[name]=engine.cell(cfg,batch,rng,reference,bodies[body],mode,attempt/name)
            assert max(r['before']['total'] for r in rows.values())-min(r['before']['total'] for r in rows.values())<=1e-7
            assert len({r['batch_sha256'] for r in rows.values()})==1
            assert len({r['order_sha256'] for r in rows.values()})==1
            atomic_json(folder/'result.json',dict(attempt=attempt.name,hashes={n:digest(attempt/n/'result.json') for n in engine.CELLS}))
            print(history,repeat,'done',flush=True)
    engine.summarize()
    print('SUMMARY',OUT/'summary.json',flush=True)


if __name__=='__main__':main()
