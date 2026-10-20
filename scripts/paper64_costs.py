"""Aggregate measured training costs and scheduler memory without splitting shared peaks."""
import argparse,json,sys,subprocess,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper64_run import *

def run(out):
    costs=[];heads=[];shared=[]
    # Keep raw-repeat summaries, and report latency quantiles over unique slides.
    raw=list(csv.DictReader((out/'reports/timing.csv').open()));grouped=collections.defaultdict(list)
    for r in raw:grouped[r['variant'],r['slide_id']].append(r)
    per=[];timekeys=[k for k in raw[0] if k.startswith('t_')]
    for (variant,sid),rr in grouped.items():
        assert len(rr)==3
        per.append({'variant':variant,'slide_id':sid,'patient_id':rr[0]['patient_id'],'n_repeats':3,'hostname':rr[0]['hostname'],
                    **{k:float(np.median([float(r[k]) for r in rr])) for k in timekeys},'aggregation':'median of three repeats, after read+compute was summed per observation'})
    csv_write(out/'reports/timing_per_slide.csv',per)
    original=out/'reports/timing_repeat_summary.csv'
    if not original.exists():atomic_bytes(original,(out/'reports/timing_summary.csv').read_bytes())
    summary=[]
    for variant in sorted({r['variant'] for r in per}):
        for key in timekeys:
            values=[r[key] for r in per if r['variant']==variant]
            summary.append({'variant':variant,'phase':key,'n':len(values),'mean_s':float(np.mean(values)),'median_s':float(np.median(values)),
                            'p95_s':float(np.quantile(values,.95)),'total_s':sum(values),'warmup_discarded':3,
                            'scope':'unique fold1 test slides; median of three repeats per slide, then distribution across slides; total_s is sum of per-slide medians, not process wall time'})
    csv_write(out/'reports/timing_summary.csv',summary)

    for fold in range(1,11):
        for seed in range(42,47):
            for task in TASKS:
                p=out/'heads'/f'fold{fold:02d}_seed{seed}'/(task+'.pt');d=checked_load(p)
                assert len(d['history'])==5 and all(h['n']==len(d['train_ids']) for h in d['history'])
                r={'fold':fold,'seed':seed,'task':task,'phase':'head training','variants':'A64 and Afour share this one head','wall_s':d['wall_s'],
                   'read_s':sum(h['t_read_s'] for h in d['history']),'compute_s':sum(h['t_compute_s'] for h in d['history']),'train_slides':len(d['train_ids']),'epochs':5,'head_sha256':file_hash(p)}
                costs.append(r);heads.append(r)
        for seed in [0,42,43,44,45,46]:
            for p,task in enumerate(TASKS):
                d=checked_load(out/path_vec(fold,seed,'train',p));groups=collections.defaultdict(list)
                for r in d['components']:groups[r['representation']].append(r)
                shared.append({'fold':fold,'seed':seed,'task':task,'phase':'joint evidence construction','wall_s':sum(r['t_read_s']+r['t_evidence_s'] for r in d['timing']),
                               'read_s':sum(r['t_read_s'] for r in d['timing']),'compute_s':sum(r['t_evidence_s'] for r in d['timing']),
                               'scope':'shared process building all representations for this seed; not a single-variant wall time'})
                for rep,rr in groups.items():
                    costs.append({'fold':fold,'seed':seed,'task':task,'phase':'train evidence construction','variants':rep,'n':len(rr),
                        'read_s':sum(r['t_read_shared_s'] for r in rr),'score_s':sum(r['t_score_shared_s'] for r in rr),
                        'selection_s':sum(r['t_selection_s'] for r in rr),'pooling_s':sum(r['t_pooling_s'] for r in rr),
                        'compute_s':sum(r['t_score_shared_s']+r['t_selection_s']+r['t_pooling_s'] for r in rr),
                        'scope':'measured components; file read and score shared across variants; not additive across variant rows'})
            for order in ORDERS:
                for stage in range(1,5):
                    for rep in (['A64','Afour'] if seed else ['B64','Bfour','MEAN']):
                        d=checked_load(out/'ridge'/f'fold{fold:02d}_seed{seed}'/order/f'{stage}_{rep}.pt')
                        costs.append({'fold':fold,'seed':seed,'order':order,'stage':stage,'phase':'readout ridge accumulate/solve','variants':rep,
                                      'accumulate_s':d['accumulate_s'],'solve_fixed_gamma_1e_minus3_s':d['solve_s']['0.001'],
                                      'all_gamma_solve_s':sum(d['solve_s'].values()),'scope':'first actual fit, excludes independent reference checks'})
        for order in ORDERS:
            for stage in range(1,5):
                d=checked_load(out/'ridge/routers'/f'fold{fold:02d}_{order}_{stage}.pt')
                costs.append({'fold':fold,'seed':0,'order':order,'stage':stage,'phase':'router fit','variants':'all internal methods share',
                              'accumulate_s':d['accumulate_s'],'solve_s':d['solve_s'],'scope':'mean evidence already constructed; shared first fit'})
    assert len(heads)==200
    csv_write(out/'reports/training_costs.csv',costs);csv_write(out/'reports/head_training.csv',heads);csv_write(out/'reports/evidence_pipeline_costs.csv',shared)
    jobs=[json.loads(x) for x in (out/'slurm/jobs.jsonl').read_text().splitlines()];ids=sorted({j['job_id'] for j in jobs});records=[]
    proc=subprocess.run(['sacct','-n','-P','-j',','.join(ids),'--format=JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList,AllocCPUS,ReqMem,AllocTRES,Start,End'],capture_output=True,text=True,check=True)
    atomic_bytes(out/'reports/slurm_accounting.txt',proc.stdout.encode());phases={j['job_id']:j.get('phase') for j in jobs}
    for line in proc.stdout.splitlines():
        v=line.split('|')
        if len(v)>=11:
            records.append(dict(zip(['job_id','state','exit_code','elapsed_s','Slurm_MaxRSS','node','allocated_cpus','requested_memory','allocated_TRES','start','end'],v[:11])));records[-1]['phase']=phases.get(v[0].split('.')[0]);records[-1]['peak_scope']='entire Slurm step/process group; never attributed to an individual variant'
    csv_write(out/'reports/memory.csv',records)
    runtimes=[]
    for p in (out/'logs').glob('*-runtime.json'):
        d=read_json(p);runtimes.append({'file':str(p.relative_to(out)),**d,'scope':'whole worker pipeline, includes cache validation/reference diagnostics; not online inference latency'})
    csv_write(out/'reports/pipeline_wall_time.csv',runtimes)
    peak=[]
    for p in (out/'manifests').glob('*_*.done.json'):
        d=read_json(p)
        if 'process_peak_rss_kb' in d:peak.append({'manifest':str(p.relative_to(out)),'process_peak_rss_kb':d['process_peak_rss_kb'],'scope':'shared worker process cumulative peak; not variant-specific'})
    csv_write(out/'reports/process_memory.csv',peak)
    write_json(out/'reports/costs_integrity.json',{'pass':True,'heads':200,'training_cost_rows':len(costs),'Slurm_rows':len(records),'all_patches_read_and_scored':True,
        'excluded_from_online':['CONCH feature extraction','integrity scan','oracle diagnostics','cached evidence lookup'],
        'timing_read_policy':'same actual file read shared by interleaved variants; per-observation read+compute summed before median/P95; never sum marginal P95s'})
    print('costs validated',len(costs),len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();setup(4);run(a.output.resolve())
