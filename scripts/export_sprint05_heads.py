"""Verify, independently recompute contrasts and export the H1 report."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,stdev

from causal_trainability import head_reset_factorial as engine
from causal_trainability.sprint03_live import atomic_json,digest
from causal_trainability.telemetry import ROOT


def main():
    gate=json.loads((engine.CAL/'summary.json').read_text())
    assert gate['passed'] and gate['identity']==engine.identity()
    for path in engine.CAL.rglob('result.json'):engine.verified(path)
    engine.summarize()
    source=engine.OUT/'summary.json';data=json.loads(source.read_text())
    rows={(r['history'],r['repeat'],r['arm']):r for r in data['rows']}
    assert len(rows)==48
    for h in range(6):
        for repeat in (0,1):
            r={arm:rows[f'breakout_cnn_s{h}',repeat,arm] for arm in engine.ARMS}
            assert r['neither']['start']['return']==r['critic']['start']['return']
            assert r['actor']['start']['return']==r['both']['start']['return']
    # Recompute directly from the four outcomes, independently of factor_effect.
    for kind in ('actor','critic','interaction'):
        groups=[]
        for h in range(6):
            values=[]
            for repeat in (0,1):
                r={arm:rows[f'breakout_cnn_s{h}',repeat,arm]['auc'] for arm in engine.ARMS}
                if kind=='actor':v=.5*(r['actor']+r['both']-r['neither']-r['critic'])
                elif kind=='critic':v=.5*(r['critic']+r['both']-r['neither']-r['actor'])
                else:v=r['both']+r['neither']-r['actor']-r['critic']
                values.append(v)
            groups.append(values)
        ds=[mean(v) for v in groups];m=mean(ds);half=3.163381449748609*stdev(ds)/math.sqrt(6)
        stored=data['effects'][kind]
        assert abs(m-stored['mean'])<1e-12
        assert all(abs(a-b)<1e-12 for a,b in zip([m-half,m+half],stored['ci975']))
        assert abs(math.sqrt(mean(stdev(v)**2 for v in groups))-stored['within_history_sd'])<1e-12
    arms={}
    for arm in engine.ARMS:
        selected=[r for r in data['rows'] if r['arm']==arm]
        arms[arm]=dict(start_return=mean(r['start']['return'] for r in selected),
            start_entropy=mean(r['start']['entropy'] for r in selected),start_value_rms=mean(r['start']['value_rms'] for r in selected),
            first_window=mean(r['windows'][0] for r in selected),windows=[mean(r['windows'][i] for r in selected) for i in range(4)],
            auc=mean(r['auc'] for r in selected),final_return=mean(r['final_return'] for r in selected))
    dest=ROOT/'research/results/sprint05-heads';dest.mkdir(parents=True,exist_ok=True)
    compact=dict(data,rows=[{k:v for k,v in row.items() if k!='curve'} for row in data['rows']],
        export=dict(source=str(source.relative_to(ROOT)),source_sha256=digest(source),omitted_fields=['rows[*].curve']))
    atomic_json(dest/'summary.json',compact)
    (dest/'calibration.json').write_bytes((engine.CAL/'summary.json').read_bytes())
    atomic_json(dest/'descriptive.json',dict(kind='descriptive_equal_weight_histories',arms=arms))
    atomic_json(dest/'manifest.json',dict(source_manifest=json.loads((engine.OUT/'manifest.json').read_text()),
        full_summary_source=str(source.relative_to(ROOT)),full_summary_sha256=digest(source),
        exported_sha256={p.name:digest(p) for p in dest.glob('*.json') if p.name!='manifest.json'},
        protocol_sha256=digest(engine.PROTOCOL),analysis_script_sha256=digest(Path(__file__)),verified_science_cells=48,
        verified_calibration_cells=len(list(engine.CAL.rglob('result.json')))))
    print(json.dumps(dict(effects=data['effects'],arms=arms),indent=2))


if __name__=='__main__':main()
