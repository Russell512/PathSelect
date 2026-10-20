"""Slurm worker, exact unit receipts and deterministic pilot cross-checks."""
import argparse,os,sys,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper64_common import *
from scripts.paper64_run import Run,checked_load,path_vec,manifest,selection

def receipt(out,spec,extra=None):
    artifacts={}
    for p in out.rglob('*.pt.done.json'):
        if 'source' in p.parts:continue
        m=read_json(p);data=p.with_suffix('').with_suffix('')
        assert data.exists() and file_hash(data)==m['sha256']
        artifacts[str(data)]=m['sha256']
    artifacts.update(extra or {})
    write_json(Path(spec['receipt']),{'pass':True,'finished':now(),'job_id':os.environ['SLURM_JOB_ID'],'spec':spec,'artifacts':artifacts})

def checks(run):
    rows=[]
    for p in range(4):
        for split in ['train','val','test']:
            d=run.getvec(p,split,0);ds,shift=run.dataset(p,split)
            for i in [0,len(ds)//2,len(ds)-1]:
                sid=ds.sids[i]
                if sid not in d['ids']:continue
                rec,_=run.read(ds,shift,i);j=d['ids'].index(sid)
                assert torch.equal(mean_norm(rec.Z),d['vectors']['MEAN'][j,0])
                for q in ([p] if split=='train' else range(4)):
                    score=zscore(text_nav_feats(rec.Z,run.text[q]['f_txt'])[:,0])
                    for k in [16,32,64,128,256]:
                        ix=select_indices(rec.Z,score,'top64',k);saved=d['indices'][f'B{k}'][j][0 if split=='train' else q]
                        assert ix.tolist()==saved
                        assert torch.equal(mean_norm(rec.Z,ix),d['vectors'][f'B{k}'][j,0 if split=='train' else q])
                rows.append({'task':p,'split':split,'id':sid,'N':len(rec.Z),'deterministic_repeat_exact':True})
    write_json(run.out/'reports/deterministic_repeats.json',rows)
    # Table 2(a)/(b) must share both subtype weights and per-slide oracle predictions.
    for order in ['paper','reverse']:
        for stage in range(1,5):
            for g in [1e-5,1e-4,1e-3,1e-2,1e-1]:
                parent=run.out/'predictions/test/fold01_seed0'
                a=checked_load(parent/f'MEAN_LIN8_g{g}'/order/f'stage{stage}.pt')
                b=checked_load(parent/f'MEAN_RDG_g{g}'/order/f'stage{stage}.pt')
                assert [r['oracle_pred'] for r in a['rows']]==[r['oracle_pred'] for r in b['rows']]
    write_json(run.out/'reports/mean_ladder_parity.json',{'pass':True,'same_subtype_state_file':True,'oracle_predictions_exact':True})

def main(spec):
    setup(4);out=Path(spec['output']);kind=spec['phase'];start=time.perf_counter()
    if kind=='tests':
        logfile=out/'logs'/f'pytest-{os.environ["SLURM_JOB_ID"]}.log'
        with logfile.open('w') as f:
            subprocess.run([sys.executable,'-m','pytest','-q','-rs'],stdout=f,stderr=subprocess.STDOUT,check=True)
        receipt(out,spec,{str(logfile):file_hash(logfile)});return
    if kind=='preflight':
        from scripts.preflight_paper64 import preflight
        from scripts.paper64_manifest import initialize
        preflight(out);initialize(out)
    elif kind in ['smoke','pilot']:
        manifest(out)
        for seed in [0,42]:
            r=Run(out,1,seed,kind=='smoke');r.build_evidence(['train','val','test']);r.evaluate('val',False);r.evaluate('test',False)
        checks(Run(out,1,0,kind=='smoke'))
    elif kind in ['prepare','test']:
        for f in spec['folds']:
            for seed in [0,42,43,44,45,46]:
                r=Run(out,f,seed)
                if kind=='prepare':r.build_evidence(['train','val']);r.evaluate('val')
                else:r.build_evidence(['test']);r.evaluate('test')
    elif kind=='select':
        from scripts.paper64_manifest import freeze
        selection(out);freeze(out)
    elif kind=='timing':
        from scripts.paper64_timing import run_timing
        run_timing(out)
    elif kind=='report':
        from scripts.paper64_report import Report
        Report(out).run()
    else:raise ValueError(kind)
    write_json(out/'logs'/f'{kind}-{os.environ["SLURM_JOB_ID"]}-runtime.json',{'wall_s':time.perf_counter()-start,'finished':now(),'hostname':socket.gethostname()})
    receipt(out,spec)
    if kind=='pilot':
        write_json(Path(spec['study_output'])/'manifests/pilot_receipt.json',read_json(Path(spec['receipt'])))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('spec',type=Path);a=ap.parse_args();main(read_json(a.spec))
