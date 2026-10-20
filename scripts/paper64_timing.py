"""Interleaved raw-feature online timing. Cached evidence is used only for parity checks."""
import argparse,resource,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper64_run import *

def tensor_bytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x,(list,tuple)):return sum(tensor_bytes(v) for v in x)
    return 0

def online(Z,rep,router,reader,run):
    t0=time.perf_counter();tp,rs=route(Z,router);t1=time.perf_counter()
    score=run.heads[tp](Z,run.text[tp]['f_txt']) if rep.startswith('A') else zscore(text_nav_feats(Z,run.text[tp]['f_txt'])[:,0]);t2=time.perf_counter()
    idx=select_indices(Z,score,'four16' if rep.endswith('four') else 'top64');t3=time.perf_counter()
    V=mean_norm(Z,idx);t4=time.perf_counter()
    pred,sc,margin=classify(V,reader,tp);t5=time.perf_counter()
    return pred,idx,V,{'router':t1-t0,'score':t2-t1,'selection':t3-t2,'pooling':t4-t3,'readout':t5-t4,'online_compute':t5-t0}

def run_timing(out,fold=1,seed=42):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    run=Run(out,fold,seed)
    for p in range(4):run.train_head(p)
    router=checked_load(out/'ridge/routers'/f'fold{fold:02d}_reverse_4.pt');readers={};reps=['A64','Afour','B64','Bfour']
    for rep in reps:
        unit=f'fold{fold:02d}_seed{seed if rep[0]=="A" else 0}'
        state=checked_load(out/'ridge'/unit/'reverse'/f'4_{rep}.pt');readers[rep]={**state,'W':state['Ws']['0.001']}
    rows=[];memory=[];models=[];start=time.perf_counter()
    with torch.no_grad():
        for p in range(4):
            ds,shift=run.dataset(p,'test')
            evs={rep:run.getvec(p,'test',seed if rep[0]=='A' else 0) for rep in reps}
            rec,_=run.read(ds,shift,0)
            for _ in range(3):
                for rep in reps:online(rec.Z,rep,router,readers[rep],run)
            for i in range(len(ds)):
                rec,tr=run.read(ds,shift,i)
                for repeat in range(3):
                    shift_order=(i+repeat)%4
                    for rep in reps[shift_order:]+reps[:shift_order]:
                        pred,idx,v,ts=online(rec.Z,rep,router,readers[rep],run)
                        tp,_=route(rec.Z,router);ev=evs[rep];j=ev['ids'].index(rec.sid)
                        assert idx.tolist()==ev['indices'][rep][j][tp]
                        assert torch.equal(v,ev['vectors'][rep][j,tp])
                        expected,_=predict(v[None],torch.tensor([tp]),readers[rep],torch.cat([d['f_txt'] for d in run.text]))
                        assert pred==int(expected[0])
                        rows.append({'fold':fold,'seed':seed if rep[0]=='A' else 0,'variant':rep,'slide_id':rec.sid,'patient_id':ds.sid2pid[rec.sid],
                            'true_task':p,'n_patches':len(rec.Z),'repeat':repeat,'hostname':socket.gethostname(),'threads':4,'device':'cpu',
                            't_read_s':tr,'t_read_compute_s':tr+ts['online_compute'],**{'t_'+k+'_s':v for k,v in ts.items()}})
    csv_write(out/'reports/timing.csv',rows);summary=[]
    for rep in reps:
        rr=[r for r in rows if r['variant']==rep]
        for key in [k for k in rr[0] if k.startswith('t_')]:
            values=[r[key] for r in rr];summary.append({'variant':rep,'phase':key,'n':len(values),'mean_s':float(np.mean(values)),'median_s':float(np.median(values)),
                'p95_s':float(np.quantile(values,.95)),'total_s':sum(values),'warmup_discarded':3,'scope':'fold1 test; three interleaved repeats; read shared, compute actual'})
        s=readers[rep];hbytes=sum(tensor_bytes(h.state_dict()) for h in run.heads.values()) if rep[0]=='A' else 0
        models.append({'variant':rep,'stage':4,'head_bytes':hbytes,'readout_S_R_W_bytes':tensor_bytes({k:s[k] for k in ['A','B','W']}),
            'router_S_R_W_bytes':tensor_bytes({k:router[k] for k in ['A','B','W']}),'texts_bytes':tensor_bytes(torch.cat([d['f_txt'] for d in run.text])),
            'note':'resident model/statistics tensor bytes; excludes audit vectors, raw features and Python allocator'})
    csv_write(out/'reports/timing_summary.csv',summary);csv_write(out/'reports/model_state_bytes.csv',models)
    write_json(out/'reports/timing_process.json',{'hostname':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],'wall_s':time.perf_counter()-start,
            'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'shared timing process; cannot assign peak separately to variants','oracle_in_timing':False})
    print('timing validated',len(rows),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);x=a.parse_args();setup(4);run_timing(x.output.resolve())
