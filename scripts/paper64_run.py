"""Frozen-source paper experiment: evidence, validation, test, and real online timing."""
from __future__ import annotations
import argparse
import csv
import io
import resource
import traceback
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper64_data import DataRun
from scripts.paper64_common import *
from selector.incremental_ridge import IncrementalRidge

GRID=[1e-5,1e-4,1e-3,1e-2,1e-1]
KS=[16,32,64,128,256]

def csv_write(path,rows):
    if not rows: raise ValueError(f'empty table {path}')
    b=io.StringIO(); w=csv.DictWriter(b,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    w.writeheader();w.writerows(rows);atomic_bytes(path,b.getvalue().encode())

def rep_names(seed):
    return ['A64','Afour'] if seed else ['MEAN']+[f'B{k}' for k in KS]+['Bfour']+[f'RANDOM{s}' for s in range(42,47)]

def path_vec(fold,seed,split,p):
    group=f'A{seed}' if seed else 'shared'
    return Path('vectors')/f'fold{fold:02d}'/group/f'{split}_{TASKS[p]}.pt'

def checked_load(path):
    m=read_json(path.with_suffix(path.suffix+'.done.json'))
    x=load_artifact(path,m['metadata'])
    if x is None: raise ValueError(f'missing {path}')
    return x

def pick(V,tasks):
    # Independent of the true task. Caller supplies router tasks for CIL, true tasks ONLY for oracle.
    return V[:,0] if V.shape[1]==1 else V[torch.arange(len(V)),tasks]

def predict(V, tasks, state, text, readout='RDG', unmasked=False):
    classes=state['classes']
    scores=aug(V)@state['W'] if readout=='RDG' else V@text[classes].T
    if unmasked:
        pred=torch.tensor(classes)[scores.argmax(1)]
    else:
        pair=torch.tensor([[classes.index(2*p),classes.index(2*p+1)] for p in tasks.tolist()])
        pred=2*tasks+scores.gather(1,pair).argmax(1)
    return pred,scores

def metrics(rows,seen):
    out={};per=[]
    for p in sorted(seen):
        rr=[r for r in rows if r['true_task']==p]
        assert rr
        d={'task':p,'n':len(rr)}
        for k,v in [('ACC','pred_label'),('Masked_ORACLE','oracle_pred'),('Masked_ROUTED_EVIDENCE','legacy_pred')]:
            d[k]=sum(r[v]==r['true_label'] for r in rr)/len(rr)
        d['TP']=sum(r['pred_task']==p for r in rr)/len(rr)
        per.append(d)
    for k in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','TP']:out[k]=float(np.mean([r[k] for r in per]))
    return {**out,'per_task':per,'n':len(rows)}

class Run(DataRun):
    def vector(self,p,split):
        ds,shift=self.dataset(p,split)
        indices=list(range(len(ds)))
        if self.smoke: indices=[next(i for i,s in enumerate(ds.sids) if ds.sid2label[s]==c) for c in [0,1]]
        ids=[ds.sids[i] for i in indices];reps=rep_names(self.seed)
        ps=[p] if split=='train' else list(range(4))
        heads={str(q):self.head_hash[q] for q in ps} if self.seed else {}
        meta={**self.own,'kind':'evidence','split':split,'task':p,'representations':reps,'head_checksums':heads,
              'ids_hash':ids_hash(ids),'selection_budgets':KS,'four16_lambda':1.5,'evidence_tasks':ps}
        def build():
            vv={k:[] for k in reps};ii={k:[] for k in reps};times=[];components=[];labels=[];patches=[]
            with torch.no_grad():
                for j,i in enumerate(indices):
                    r,tr=self.read(ds,shift,i);labels.append(r.label);patches.append(len(r.Z));tic=time.perf_counter()
                    vals={};inds={}
                    if not self.seed:
                        vals['MEAN']=mean_norm(r.Z)[None];inds['MEAN']=[list(range(len(r.Z)))]
                        for seed in range(42,47):
                            idx=random_indices(len(r.Z),seed,self.fold,r.sid)
                            vals[f'RANDOM{seed}']=mean_norm(r.Z,idx)[None];inds[f'RANDOM{seed}']=[idx.tolist()]
                    for q in ps:
                        ts=time.perf_counter()
                        score=self.heads[q](r.Z,self.text[q]['f_txt']) if self.seed else zscore(text_nav_feats(r.Z,self.text[q]['f_txt'])[:,0])
                        score_s=time.perf_counter()-ts
                        for key,k,mode in ([('A64',64,'top64'),('Afour',64,'four16')] if self.seed else [(f'B{k}',k,'top64') for k in KS]+[('Bfour',64,'four16')]):
                            ts=time.perf_counter();idx=select_indices(r.Z,score,mode,k);selection_s=time.perf_counter()-ts
                            ts=time.perf_counter();ev=mean_norm(r.Z,idx);pooling_s=time.perf_counter()-ts
                            vals.setdefault(key,[]).append(ev);inds.setdefault(key,[]).append(idx.tolist())
                            components.append({'slide_id':r.sid,'representation':key,'evidence_task':q,'t_score_shared_s':score_s,'t_selection_s':selection_s,'t_pooling_s':pooling_s,'t_read_shared_s':tr})
                    for key in reps:
                        vv[key].append(vals[key] if isinstance(vals[key],torch.Tensor) else torch.stack(vals[key]));ii[key].append(inds[key])
                    times.append({'slide_id':r.sid,'t_read_s':tr,'t_evidence_s':time.perf_counter()-tic})
                    self.beat('evidence',split=split,task=p,slide=j,n=len(ids))
            return {'ids':ids,'patients':[ds.sid2pid[s] for s in ids],'labels':torch.tensor(labels),'tasks':torch.full((len(ids),),p),
                    'vectors':{k:torch.stack(vv[k]) for k in reps},'indices':ii,'n_patches':patches,'timing':times,'components':components,'head_checksums':heads}
        def valid(x):
            assert x['ids']==ids and x['patients']==[ds.sid2pid[s] for s in ids]
            assert x['labels'].tolist()==[int(ds.sid2label[s])+shift for s in ids]
            assert x['head_checksums']==heads
            for key in reps:
                nt=len(ps) if key[0] in 'AB' else 1
                assert x['vectors'][key].shape==(len(ids),nt,512)
                for n,selected in zip(x['n_patches'],x['indices'][key]):
                    assert len(selected)==nt
                    k=n if key=='MEAN' else (int(key[1:]) if key.startswith('B') and key[1:].isdigit() else 64)
                    for ix in selected:
                        assert len(ix)==min(n,k) and len(set(ix))==len(ix) and min(ix)>=0 and max(ix)<n
        return self.artifact(path_vec(self.fold,self.seed,split,p),meta,build,valid)

    def build_evidence(self,splits):
        if self.seed:
            for p in range(4):self.train_head(p)
        for split in splits:
            for p in range(4):self.vector(p,split)
        write_json(self.out/'manifests'/f'evidence_{self.unit}_{"_".join(splits)}.json',{'metadata':self.own,'artifacts':self.artifacts,'finished':now()})

    def getvec(self,p,split,seed=None):
        path=self.out/path_vec(self.fold,self.seed if seed is None else seed,split,p)
        meta=read_json(path.with_suffix(path.suffix+'.done.json'))['metadata']
        for key in ['source_fingerprint','environment_hash','data_hash','text_hash','protocol_hash','namespace','fold']:
            if meta[key]!=self.base[key]:raise ValueError(f'evidence provenance mismatch {key}: {path}')
        return checked_load(path)

    def fit(self):
        states={};rt={};checks=[]
        train=[self.getvec(p,'train') for p in range(4)]
        validation=[self.getvec(p,'val') for p in range(4)]
        shared=train if not self.seed else [self.getvec(p,'train',0) for p in range(4)]
        for order,names in ORDERS.items():
            router=IncrementalRidge(1e-3);readers={r:Readout() for r in rep_names(self.seed)};seen=[]
            for stage,task in enumerate(names,1):
                p=TASKS.index(task);seen.append(p);d=train[p]
                t=time.perf_counter();router.add_task(p,shared[p]['vectors']['MEAN'][:,0]);ta=time.perf_counter()-t
                t=time.perf_counter();rw=router.solve();ts=time.perf_counter()-t
                rs={'A':router.A.clone(),'B':torch.stack(router.b,1),'W':rw,'tasks':list(router.task_ids),'accumulate_s':ta,'solve_s':ts}
                rm={**self.base,'kind':'router','order':order,'stage':stage,'gamma_tp':1e-3}
                def vr(x):
                    assert x['tasks']==rs['tasks']
                    for k in ['A','B','W']:torch.testing.assert_close(x[k],rs[k],atol=1e-8,rtol=1e-7)
                rt[order,stage]=self.artifact(Path('ridge/routers')/f'fold{self.fold:02d}_{order}_{stage}.pt',rm,lambda:rs,vr)
                for rep,reader in readers.items():
                    t=time.perf_counter();reader.add_task(p,d['vectors'][rep][:,0],d['labels']);ta=time.perf_counter()-t
                    classes=sorted(reader.cols);B=torch.stack([reader.cols[c] for c in classes],1)
                    gammas=GRID if rep in ['MEAN','B64','A64'] else [1e-3]
                    Ws={};tt={}
                    for g in gammas:
                        t=time.perf_counter();Ws[str(g)]=torch.linalg.solve(reader.A+g*torch.eye(513,dtype=torch.float64),B);tt[str(g)]=time.perf_counter()-t
                    state={'A':reader.A.clone(),'B':B,'Ws':Ws,'classes':classes,'tasks':list(seen),'accumulate_s':ta,'solve_s':tt}
                    # Independent batch reference on the same stage's train-only evidence.
                    X=aug(torch.cat([train[q]['vectors'][rep][:,0] for q in seen]));y=torch.cat([train[q]['labels'] for q in seen])
                    Ar=X.T@X;Br=torch.stack([X[y==c].sum(0) for c in classes],1)
                    torch.testing.assert_close(state['A'],Ar,atol=1e-8,rtol=1e-7);torch.testing.assert_close(B,Br,atol=1e-8,rtol=1e-7)
                    for g in gammas:
                        wr=torch.linalg.solve(Ar+g*torch.eye(513,dtype=torch.float64),Br)
                        # Low-gamma ill-conditioning magnifies roundoff in weights; validate
                        # both backward error and every train/validation decision independently.
                        w=Ws[str(g)]
                        torch.testing.assert_close(w,wr,atol=1e-6,rtol=1e-7)
                        residual=(state['A']+g*torch.eye(513,dtype=torch.float64))@w-B
                        assert float(residual.abs().max()) < 1e-8
                        refs=[X]+[aug(validation[q]['vectors'][rep].reshape(-1,512)) for q in seen]
                        for ref in refs:
                            left,right=ref@w,ref@wr
                            torch.testing.assert_close(left,right,atol=1e-8,rtol=1e-7)
                            assert torch.equal(left.argmax(1),right.argmax(1)), 'batch-reference near tie'
                            for q in seen:
                                pair=[classes.index(2*q),classes.index(2*q+1)]
                                assert torch.equal(left[:,pair].argmax(1),right[:,pair].argmax(1)), 'pair near tie'
                    meta={**self.own,'kind':'ridge','representation':rep,'gammas':gammas,'order':order,'stage':stage,
                          'train_hashes':[file_hash(self.out/path_vec(self.fold,self.seed,'train',q)) for q in seen]}
                    def valid(x):
                        assert x['classes']==classes and x['tasks']==seen
                        for k in ['A','B']:torch.testing.assert_close(x[k],state[k],atol=1e-8,rtol=1e-7)
                        for g in gammas:torch.testing.assert_close(x['Ws'][str(g)],Ws[str(g)],atol=1e-8,rtol=1e-7)
                    states[order,stage,rep]=self.artifact(Path('ridge')/self.unit/order/f'{stage}_{rep}.pt',meta,lambda:state,valid)
                    checks.append({'order':order,'stage':stage,'rep':rep,'batch_reference':True})
        write_json(self.out/'reports'/f'{self.unit}_ridge_reference.json',checks)
        return states,rt

    def evaluate(self,split,formal=True):
        if split=='test' and formal and not (self.out/'protocol/selection.json').exists():raise ValueError('freeze gamma selection before test')
        states,routers=self.fit();reps=rep_names(self.seed);summaries=[];last={}
        data=[self.getvec(p,split) for p in range(4)]
        shared=data if not self.seed else [self.getvec(p,split,0) for p in range(4)]
        F=torch.cat([t['f_txt'] for t in self.text])
        for order,names in ORDERS.items():
            for stage in range(1,5):
                seen=[TASKS.index(t) for t in names[:stage]];ps=sorted(seen)
                ids=sum([data[p]['ids'] for p in ps],[]);pids=sum([data[p]['patients'] for p in ps],[])
                y=torch.cat([data[p]['labels'] for p in ps]);true=torch.cat([data[p]['tasks'] for p in ps])
                mean=torch.cat([shared[p]['vectors']['MEAN'][:,0] for p in ps]);router=routers[order,stage]
                rs=aug(mean)@router['W'];cols=sorted(router['tasks']);rs=rs[:,[router['tasks'].index(p) for p in cols]]
                predicted_task=torch.tensor(cols)[rs.argmax(1)]
                for rep in reps:
                    if split=='val' and rep not in ['MEAN','A64','B64']:continue
                    V=torch.cat([data[p]['vectors'][rep] for p in ps]);cilV=pick(V,predicted_task);oracleV=pick(V,true)
                    state=states[order,stage,rep]
                    modes=['RDG','LIN8'] if rep=='MEAN' else ['RDG','COS'] if rep in ['A64','B64'] else ['RDG']
                    for mode in modes:
                        gammas=list(state['Ws']) if mode!='COS' else ['none']
                        for gs in gammas:
                            cfgid=f'{rep}_{mode}_g{gs}';gamma=None if gs=='none' else float(gs)
                            st={**state,'W':state['Ws'][str(1e-3) if gs=='none' else gs]}
                            pred,scores=predict(cilV,predicted_task,st,F,'COS' if mode=='COS' else 'RDG',mode=='LIN8')
                            oracle,oscores=predict(oracleV,true,st,F,'COS' if mode=='COS' else 'RDG')
                            legacy,lscores=predict(cilV,true,st,F,'COS' if mode=='COS' else 'RDG')
                            if mode=='LIN8':pred,scores=predict(oracleV,true,st,F,'RDG',True)
                            meta={**self.own,'kind':'predictions','split':split,'experiment_id':cfgid,'order':order,'stage':stage,
                                  'gamma_wp':gamma,'gamma_tp':1e-3,'representation':rep,'ids_hash':ids_hash(ids),
                                  'evidence_files':[str(path_vec(self.fold,self.seed,split,p)) for p in ps],
                                  'evidence_hashes':[file_hash(self.out/path_vec(self.fold,self.seed,split,p)) for p in ps],
                                  'ridge_file':str(Path('ridge')/self.unit/order/f'{stage}_{rep}.pt'),
                                  'ridge_hash':file_hash(self.out/'ridge'/self.unit/order/f'{stage}_{rep}.pt')}
                            config_hash=digest({k:v for k,v in meta.items() if k not in ['kind','split','ids_hash','evidence_files','evidence_hashes']})
                            rows=[]
                            seed=int(rep[6:]) if rep.startswith('RANDOM') else self.seed
                            for i,sid in enumerate(ids):
                                rows.append({'experiment_id':cfgid,'config_hash':config_hash,'fold':self.fold,'seed':seed,
                                    'order':order,'stage':stage,'slide_id':sid,'patient_id':pids[i],'true_task':int(true[i]),'true_label':int(y[i]),
                                    'pred_task':int(predicted_task[i]),'pred_label':int(pred[i]),'oracle_pred':int(oracle[i]),'legacy_pred':int(legacy[i]),
                                    'scores':scores[i].tolist(),'oracle_scores':oscores[i].tolist(),'class_columns':state['classes'],
                                    'router_scores':rs[i].tolist(),'router_tasks':cols,'evidence_reference':meta['evidence_files'][ps.index(int(true[i]))]})
                            payload={'rows':rows,'metrics':metrics(rows,seen),'evidence_hashes':meta['evidence_hashes']}
                            def valid(x):
                                assert [r['slide_id'] for r in x['rows']]==ids and x['metrics']==metrics(x['rows'],seen)
                                assert [r['pred_label'] for r in x['rows']]==pred.tolist()
                                assert [r['oracle_pred'] for r in x['rows']]==oracle.tolist()
                                if rep=='MEAN': assert torch.equal(oracle,legacy)
                            path=Path('predictions')/split/self.unit/cfgid/order/f'stage{stage}.pt'
                            value=self.artifact(path,meta,lambda:payload,valid)
                            summaries.append({'experiment_id':cfgid,'fold':self.fold,'seed':seed,'order':order,'stage':stage,'split':split,
                                              'gamma':gamma,'prediction_file':str(path),'sha256':file_hash(self.out/path),**value['metrics']})
                            if stage==4:last[order,cfgid]=value['rows']
        parity={}
        for cfgid in set(k[1] for k in last):
            a,b=last['paper',cfgid],last['reverse',cfgid]
            assert [r['slide_id'] for r in a]==[r['slide_id'] for r in b]
            diff=[];err=0.
            for x,z in zip(a,b):
                for k in ['scores','oracle_scores','router_scores']:
                    np.testing.assert_allclose(x[k],z[k],atol=1e-8 if 'COS' not in cfgid else 1e-6,rtol=1e-7)
                    err=max(err,max(abs(v-w) for v,w in zip(x[k],z[k])))
                if any(x[k]!=z[k] for k in ['pred_label','oracle_pred','legacy_pred','pred_task']):diff.append(x['slide_id'])
            parity[cfgid]={'prediction_disagreements':diff,'max_score_difference':err}
        write_json(self.out/'reports'/f'{self.unit}_{split}_order_parity.json',parity)
        if any(v['prediction_disagreements'] for v in parity.values()):raise ValueError('near tie order disagreement: investigate')
        save_artifact(self.out/'metrics'/f'{self.unit}_{split}.pt',summaries,{**self.own,'split':split,'kind':'metrics'})
        write_json(self.out/'manifests'/f'{self.unit}_{split}.done.json',{'metadata':self.own,'rows':len(summaries),'artifacts':self.artifacts,'finished':now(),
            'process_peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_scope':'whole process, not variant-specific'})
        return summaries

def selection(out):
    targets={'MEAN':'MEAN_RDG','B64':'B64_RDG','A64':'A64_RDG'};records=[];selected={}
    for rep,prefix in targets.items():
        scores={g:[] for g in GRID}
        for f in range(1,11):
            for seed in (range(42,47) if rep=='A64' else [0]):
                rows=checked_load(out/'metrics'/f'fold{f:02d}_seed{seed}_val.pt')
                for g in GRID:
                    rr=[r for r in rows if r['experiment_id']==f'{prefix}_g{g}'];assert len(rr)==8
                    objective=float(np.mean([r['Masked_ORACLE'] for r in rr]));scores[g].append(objective)
                    records.append({'representation':rep,'fold':f,'seed':seed,'gamma':g,'validation_objective':objective})
                    if rep=='MEAN':
                        ll=[r for r in rows if r['experiment_id']==f'MEAN_LIN8_g{g}'];assert len(ll)==8
                        assert all(a['Masked_ORACLE']==b['Masked_ORACLE'] for a,b in zip(rr,ll))
        means={g:float(np.mean(v)) for g,v in scores.items()};best=max(GRID,key=lambda g:(means[g],g))
        selected[rep]={'gamma':best,'objectives':means,'boundary':best in [GRID[0],GRID[-1]]}
    payload={'selected':selected,'rule':'validation only; tasks/stages/orders/folds/seeds equal; exact ties larger gamma','frozen_at':now(),
             'validation_files':{str(p.relative_to(out)):file_hash(p) for p in sorted((out/'metrics').glob('*_val.pt'))}}
    path=out/'protocol/selection.json'
    if path.exists():
        old=read_json(path);assert old['selected']==payload['selected'] and old['validation_files']==payload['validation_files']
    else:write_json(path,payload)
    csv_write(out/'reports/gamma_validation.csv',records)
    manifest(out,selected)
    print(json.dumps(selected),flush=True)

def manifest(out,selected=None):
    g=lambda r: selected[r]['gamma'] if selected else 'VALIDATION_PENDING'
    aliases={'LIN8_MEAN':f'MEAN_LIN8_g{g("MEAN")}','ROUTED_MEAN':f'MEAN_RDG_g{g("MEAN")}',
             'B_TOP64_RIDGE':f'B64_RDG_g{g("B64")}','B_TOP64_COS':'B64_COS_gnone',
             'A_TOP64_RIDGE':f'A64_RDG_g{g("A64")}','A_TOP64_COS':'A64_COS_gnone',
             'B_TOP64_FIXED':'B64_RDG_g0.001','B_FOUR16_FIXED':'Bfour_RDG_g0.001',
             'A_TOP64_FIXED':'A64_RDG_g0.001','A_FOUR16_FIXED':'Afour_RDG_g0.001',
             'ROUTED_MEAN_FIXED':'MEAN_RDG_g0.001','RANDOM64_RIDGE_FIXED':[f'RANDOM{s}_RDG_g0.001' for s in range(42,47)]}
    mapping={'Table1_Ridge':'LIN8_MEAN','Table1_PathSelect':'B_TOP64_RIDGE','Table1_residual':'A_TOP64_RIDGE',
             'Table2_a':'LIN8_MEAN','Table2_b':'ROUTED_MEAN','Table2_c':'B_TOP64_RIDGE',
             'Table2_text_ridge':'B_TOP64_RIDGE','Table2_text_cosine':'B_TOP64_COS','Table2_residual_ridge':'A_TOP64_RIDGE','Table2_residual_cosine':'A_TOP64_COS'}
    write_json(out/'protocol'/('expanded_manifest.json' if selected else 'expected_manifest.json'),
        {'folds':list(range(1,11)),'A_random_seeds':list(range(42,47)),'A_heads':200,'orders':['paper','reverse'],'stages':[1,2,3,4],
         'aliases':aliases,'table_map':mapping,'budget_K':KS,'gamma_grid':GRID,'deterministic_statistical_repeats':1,'external_required':True})
    write_json(out/'protocol/table_map.json',mapping)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--fold',type=int,default=1)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--phase',choices=['manifest','prepare','test','select','pilot'],required=True)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args();setup(4);out=a.output.resolve()
    if a.phase=='manifest':manifest(out)
    elif a.phase=='select':selection(out)
    else:
        start=time.perf_counter()
        try:
            r=Run(out,a.fold,a.seed,a.smoke)
            if a.phase=='prepare':r.build_evidence(['train','val']);r.evaluate('val')
            elif a.phase=='test':r.build_evidence(['test']);r.evaluate('test')
            else:r.build_evidence(['train','val','test']);r.evaluate('val',False);r.evaluate('test',False)
            write_json(out/'logs'/f'{r.unit}_{a.phase}_runtime.json',{'wall_s':time.perf_counter()-start,'finished':now(),'hostname':socket.gethostname(),
                'job_id':os.environ['SLURM_JOB_ID'],'process_peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_scope':'whole process'})
        except Exception:
            atomic_bytes(out/'logs'/f'FAILED_{a.phase}_fold{a.fold}_seed{a.seed}_{time.time_ns()}.txt',traceback.format_exc().encode());raise
