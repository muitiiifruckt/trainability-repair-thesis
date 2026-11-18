"""Verify and export the matched-policy comparison without large training curves."""
import json
import math
from pathlib import Path
from statistics import mean,stdev

from causal_trainability import behavior_match as engine
from causal_trainability.sprint03_live import atomic_json,digest
from causal_trainability.telemetry import ROOT


def main():
    engine.check_pilot()
    gate=json.loads((engine.CAL/'summary.json').read_text())
    assert gate['passed'] and gate['identity']==engine.identity()
    calibration=list(engine.CAL.rglob('result.json'));assert len(calibration)==24
    for path in calibration:engine.verified(path)
    engine.summarize()
    source=engine.OUT/'summary.json';data=json.loads(source.read_text())
    rows={(r['history'],r['repeat'],r['arm']):r for r in data['rows']};assert len(rows)==36
    ds=[];groups=[]
    for h in range(6):
        values=[]
        for repeat in (0,1):
            a=rows[f'breakout_cnn_s{h}',repeat,'matched'];b=rows[f'breakout_cnn_s{h}',repeat,'reset']
            assert a['first']==b['first'] and a['start']['return']==b['start']['return']
            # Recompute AUC directly from episode windows, not stored auc fields.
            windows=[]
            for r in (a,b):
                w=[]
                for i in range(4):
                    outcomes=[v for point in r['curve'] if i*32768<point['env_steps']<=(i+1)*32768 for v in point['returns']]
                    assert outcomes;w.append(mean(outcomes))
                windows.append(mean(w))
            values.append(windows[0]-windows[1])
        groups.append(values);ds.append(mean(values))
    m=mean(ds);se=stdev(ds)/math.sqrt(6);p=data['primary']
    assert abs(m-p['mean'])<1e-12
    for key,t in [('ci90',2.0150483733330233),('ci95',2.570581835636314)]:
        assert all(abs(a-b)<1e-12 for a,b in zip([m-t*se,m+t*se],p[key]))
    assert abs(math.sqrt(mean(stdev(v)**2 for v in groups))-p['within_history_sd'])<1e-12
    arms={}
    for arm in engine.ARMS:
        selected=[r for r in data['rows'] if r['arm']==arm]
        assert all(r['anchors_unchanged'] and r['resume_bit_equal'] and r['global_rng_unchanged'] for r in selected)
        arms[arm]=dict(start_return=mean(r['start']['return'] for r in selected),start_entropy=mean(r['start']['entropy'] for r in selected),
            windows=[mean(r['windows'][i] for r in selected) for i in range(4)],auc=mean(r['auc'] for r in selected),
            final_return=mean(r['final_return'] for r in selected))
    out=ROOT/'research/results/sprint06-match';out.mkdir(parents=True,exist_ok=True)
    compact=dict(data,rows=[{k:v for k,v in row.items() if k!='curve'} for row in data['rows']],
        export=dict(source=str(source.relative_to(ROOT)),source_sha256=digest(source),omitted_fields=['rows[*].curve']))
    atomic_json(out/'summary.json',compact)
    (out/'calibration.json').write_bytes((engine.CAL/'summary.json').read_bytes())
    (out/'pilot.json').write_bytes((engine.PILOT/'summary.json').read_bytes())
    atomic_json(out/'descriptive.json',dict(kind='descriptive_equal_weight_histories',arms=arms))
    atomic_json(out/'manifest.json',dict(source_manifest=json.loads((engine.OUT/'manifest.json').read_text()),
        exported_sha256={p.name:digest(p) for p in out.glob('*.json') if p.name!='manifest.json'},
        full_summary_source=str(source.relative_to(ROOT)),full_summary_sha256=digest(source),
        analysis_script_sha256=digest(Path(__file__)),verified_science_cells=36,verified_calibration_cells=24,verified_pilot_cells=6))
    print(json.dumps(dict(primary=data['primary'],arms=arms),indent=2))


if __name__=='__main__':main()
