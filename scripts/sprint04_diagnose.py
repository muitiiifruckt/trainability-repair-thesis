"""Read-only descriptive analysis of sealed Sprint 03 outputs."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'runs/causal-trainability/sprint03_live_20261109'
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    records, inputs = [], {'summary.json': sha(source / 'summary.json')}
    for relative, expected in summary['result_hashes'].items():
        path = source / relative
        assert sha(path) == expected, relative
        row = json.loads(path.read_text(encoding='utf-8'))
        attempt = path.parent / row['attempt']
        assert sha(attempt / 'result.json') == row['attempt_result_sha256']
        assert sha(attempt / 'artifacts.jsonl') == row['artifacts_sha256']
        artifacts = [json.loads(s) for s in (attempt / 'artifacts.jsonl').read_text().splitlines()]
        tele = [a for a in artifacts if a['kind'] == 'telemetry']
        assert len(tele) == 1
        tp = attempt / tele[0]['path']
        assert sha(tp) == tele[0]['sha256']
        first = json.loads(tp.read_text().splitlines()[0])
        start = row['start']
        records.append(dict(history=row['history'], arm=row['arm'], repeat=row['repeat'],
            start=start, gates=dict(policy=start['policy_tv_vs_fresh'] <= .02,
            entropy=abs(start['entropy_delta_vs_fresh']) <= .02,
            critic=start['value_difference_rms_vs_fresh'] <= .1),
            first_window=row['windows'][0], late_auc=row['late_auc'], final_return=row['final_return'],
            first_telemetry=dict(update=first['update'], metrics=first['metrics'], layers=first['layers'])))
        inputs[relative] = expected
        inputs[str(tp.relative_to(source))] = tele[0]['sha256']
    arms = {}
    for arm in sorted({r['arm'] for r in records}):
        # Equal paired repetitions for every arm; repetitions are averaged within history first.
        selected = [r for r in records if r['arm'] == arm and r['repeat'] < 2]
        by_history = {}
        for h in sorted({r['history'] for r in selected}):
            rs = [r for r in selected if r['history'] == h]
            by_history[h] = dict(start_return=mean(r['start']['return'] for r in rs),
                policy_tv=mean(r['start']['policy_tv_vs_fresh'] for r in rs),
                entropy_delta=mean(r['start']['entropy_delta_vs_fresh'] for r in rs),
                critic_rms=mean(r['start']['value_difference_rms_vs_fresh'] for r in rs),
                first_window=mean(r['first_window'] for r in rs),
                late_auc=mean(r['late_auc'] for r in rs),
                first_logged_entropy=mean(r['first_telemetry']['metrics']['entropy'] for r in rs),
                gates={k: all(r['gates'][k] for r in rs) for k in ('policy','entropy','critic')})
        arms[arm] = dict(histories=by_history,
            means={k: mean(v[k] for v in by_history.values()) for k in
                ('start_return','policy_tv','entropy_delta','critic_rms','first_window','late_auc','first_logged_entropy')},
            gate_pass_histories={k:sum(v['gates'][k] for v in by_history.values()) for k in ('policy','entropy','critic')})
    result = dict(schema_version=1, kind='exploratory_existing_histories',
        independent_histories=6, paired_repeats=[0,1], inputs_sha256=inputs, arms=arms, records=records,
        limitation='First available telemetry is after iteration 16; no first-minibatch causal inference.')
    out = ROOT / 'research/results/sprint04/diagnosis.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')
    for arm, row in arms.items():
        print(arm, json.dumps(dict(means=row['means'], gates=row['gate_pass_histories'])))


if __name__ == '__main__':
    main()
