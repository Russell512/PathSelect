"""Reconstruct paper tables and paired uncertainty from checked predictions."""
from __future__ import annotations
import argparse,collections,csv,io,json,math,sys,statistics,subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper64_common import *
from scripts.paper64_run import checked_load,csv_write,metrics,path_vec

FIELDS={'ACC':'pred_label','Masked_ORACLE':'oracle_pred','Masked_ROUTED_EVIDENCE':'legacy_pred'}

def longitudinal(stages,order):
    pos=[TASKS.index(t) for t in ORDERS[order]]
    R=[{x['task']:x['ACC'] for x in s['per_task']} for s in stages]
    return {**{k:stages[-1][k] for k in FIELDS},'TP':stages[-1]['TP'],
        'Forgetting':float(np.mean([max(R[t][pos[j]] for t in range(j,3))-R[3][pos[j]] for j in range(3)])),
        'BWT':float(np.mean([R[3][pos[j]]-R[j][pos[j]] for j in range(3)])),
        'AIA':float(np.mean([x['ACC'] for x in stages])),
        'R':[[R[t].get(p) for p in pos] for t in range(4)]}

def sign(a,b,lower=False):
    d=np.asarray(a)-np.asarray(b);w=int(np.sum(d< -1e-12) if lower else np.sum(d>1e-12));l=int(np.sum(d>1e-12) if lower else np.sum(d< -1e-12));t=len(d)-w-l
    return {'delta_f':json.dumps(d.tolist()),'mean_difference_fraction':float(d.mean()),'mean_difference_pp':float(d.mean()*100),'wins':w,'losses':l,'ties':t,
            'n_folds':len(d),'n_nonties':w+l,'p_raw':float(binomtest(w,w+l,.5).pvalue) if w+l else 1.0}

def render_table(out,name,rows,columns=None,caption=''):
    csv_write(out/(name+'.csv'),rows);columns=columns or list(rows[0])
    def val(x):
        if x is None:return '—'
        if isinstance(x,float):return f'{x:.6g}'
        return str(x)
    md='| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'
    md+=''.join('| '+' | '.join(val(r.get(c)).replace('|','/') for c in columns)+' |\n' for r in rows)
    atomic_bytes(out/(name+'.md'),(md+'\n'+caption+'\n').encode())
    esc=lambda x:val(x).replace('_',r'\_').replace('%',r'\%').replace('±',r'$\pm$').replace('—','--')
    tex='\\begin{table}\n\\centering\n\\caption{'+esc(caption)+'}\n\\begin{tabular}{'+'l'*len(columns)+'}\n\\hline\n'+' & '.join(esc(c) for c in columns)+r' \\'+'\n'
    tex+=''.join(' & '.join(esc(r.get(c)) for c in columns)+r' \\'+'\n' for r in rows)+'\\hline\n\\end{tabular}\n\\end{table}\n'
    atomic_bytes(out/(name+'.tex'),tex.encode())

class Report:
    def __init__(self,out):
        self.out=out;self.rep=out/'reports';self.man=read_json(out/'manifests/data.json');self.exp=read_json(out/'protocol/expanded_manifest.json')
        self.host=read_json(out/'environment/compute.json');self.alias=self.exp['aliases'];self.rows={};self.summary={};self.integrity=[];self.stage=[];self.perseed=[];self.fold=[]
        self.alias['A_TOP64_RIDGE_3SEED']=self.alias['A_TOP64_RIDGE'];self.alias['A_TOP64_COS_3SEED']=self.alias['A_TOP64_COS'];self.alias['A_TOP64_FIXED_3SEED']=self.alias['A_TOP64_FIXED'];self.alias['A_FOUR16_FIXED_3SEED']=self.alias['A_FOUR16_FIXED']
        for k in [16,32,128,256]:self.alias[f'B_TOP{k}_FIXED']=f'B{k}_RDG_g0.001'
        for g in [1e-5,1e-4,1e-3,1e-2,1e-1]:self.alias[f'B_TOP64_GAMMA_{g}']=f'B64_RDG_g{g}'
        self.inputs={};self.last={};self.external=False;self.R=[]

    def verify(self,path,fold,seed,order,stage,experiment):
        meta=read_json(path.with_suffix('.pt.done.json'))['metadata']
        for k,v in [('source_fingerprint',self.host['source_fingerprint']),('environment_hash',self.host['environment_hash']),('data_hash',self.man['data_hash']),('namespace',self.out.name)]:
            assert meta[k]==v,(path,k)
        d=checked_load(path);rr=d['rows'];seen=[TASKS.index(t) for t in ORDERS[order][:stage]]
        expected={r['slide_id']:r for p in seen for r in self.man['cells'][f'{fold}|{TASKS[p]}|test']['records']}
        assert len(rr)==len(expected) and set(r['slide_id'] for r in rr)==set(expected)
        classes=sorted(c for p in seen for c in [2*p,2*p+1]);rebuild=[]
        for r in rr:
            ref=expected[r['slide_id']];assert r['true_label']==ref['global_label'] and r['patient_id']==ref['patient_id']
            assert r['class_columns']==classes and r['pred_task'] in seen
            assert r['fold']==fold and r['stage']==stage and r['order']==order and r['experiment_id']==experiment
            assert r['true_task']==ref['global_label']//2
            expected_seed=int(experiment.split('_')[0][6:]) if experiment.startswith('RANDOM') else seed
            assert r['seed']==expected_seed
            rs=np.array(r['router_scores']);assert r['pred_task']==r['router_tasks'][int(rs.argmax())]
            a=np.asarray(r['scores']);b=np.asarray(r['oracle_scores']);assert np.isfinite(a).all() and np.isfinite(b).all()
            tp=r['pred_task'];true=r['true_task'];pair=[classes.index(2*tp),classes.index(2*tp+1)];oracle=[classes.index(2*true),classes.index(2*true+1)]
            pred=classes[int(a.argmax())] if '_LIN8_' in experiment else 2*tp+int(a[pair].argmax())
            assert pred==r['pred_label'] and 2*true+int(b[oracle].argmax())==r['oracle_pred'] and 2*true+int(a[oracle].argmax())==r['legacy_pred']
        met=metrics(rr,seen);assert met==d['metrics'];self.inputs[str(path.relative_to(self.out))]=file_hash(path)
        return rr,met

    def load(self):
        cache={}
        for name,configs in self.alias.items():
            configs=configs if isinstance(configs,list) else [configs]
            seeds=[42,43,44] if name.endswith('_3SEED') else list(range(42,47)) if name.startswith('A_') else [0]
            for order in ['paper','reverse']:
                for f in range(1,11):
                    acc=[];last=[];stage_acc=[]
                    for seed in seeds:
                        for cfg in configs:
                            key=f,seed,cfg,order
                            if key not in cache:
                                ms=[];rr4=None
                                for st in range(1,5):
                                    path=self.out/'predictions/test'/f'fold{f:02d}_seed{seed}'/cfg/order/f'stage{st}.pt'
                                    rr,m=self.verify(path,f,seed,order,st,cfg);ms.append(m)
                                    if st==4:rr4=rr
                                cache[key]=(longitudinal(ms,order),rr4,ms)
                            lg,rr,ms=cache[key]
                            self.R.append({'experiment':name,'config':cfg,'fold':f,'seed':seed,'order':order,'task_columns':ORDERS[order],'R':lg['R']})
                            actual_seed=int(cfg.split('_')[0][6:]) if cfg.startswith('RANDOM') else seed
                            self.perseed.append({'experiment':name,'config':cfg,'fold':f,'seed':actual_seed,'order':order,**{k:v for k,v in lg.items() if k!='R'}})
                            acc.append(lg);last.append(rr);stage_acc.append(ms)
                    vals={k:float(np.mean([d[k] for d in acc])) for k in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','TP','Forgetting','BWT','AIA']}
                    self.fold.append({'experiment':name,'fold':f,'order':order,'seeds_averaged':len(acc),**vals})
                    self.summary[name,order,f]=vals;self.last[name,order,f]=last
                    for st in range(4):
                        self.stage.append({'experiment':name,'order':order,'fold':f,'stage':st+1,**{k:float(np.mean([m[st][k] for m in stage_acc])) for k in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','TP']}})
            self.integrity.append({'experiment':name,'orders':2,'folds':10,'stages':4,'status':'VALIDATED','alias_reference':json.dumps(configs),'statistical_n':10})
        self.load_external()
        for order in ['paper','reverse']:
            for f in range(1,11):
                for stage in range(1,5):
                    paths=[self.out/'predictions/test'/f'fold{f:02d}_seed0'/self.alias[n]/order/f'stage{stage}.pt' for n in ['LIN8_MEAN','ROUTED_MEAN']]
                    a,b=[checked_load(p) for p in paths];ma,mb=[read_json(p.with_suffix('.pt.done.json'))['metadata'] for p in paths]
                    assert ma['ridge_file']==mb['ridge_file'] and ma['ridge_hash']==mb['ridge_hash']
                    assert [r['oracle_pred'] for r in a['rows']]==[r['oracle_pred'] for r in b['rows']]
        write_json(self.rep/'R_matrices.json',self.R)
        for name in self.alias:
            for f in range(1,11):
                a,b=self.last[name,'paper',f],self.last[name,'reverse',f]
                for ra,rb in zip(a,b):
                    assert [(r['slide_id'],r['pred_label'],r['oracle_pred'],r['legacy_pred']) for r in ra]==[(r['slide_id'],r['pred_label'],r['oracle_pred'],r['legacy_pred']) for r in rb]
        csv_write(self.rep/'per_fold.csv',[{**r,**{k+'_percent':100*r[k] for k in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','Forgetting','BWT','AIA']}} for r in self.fold])
        csv_write(self.rep/'per_seed_fold.csv',self.perseed);csv_write(self.rep/'stage_curves.csv',self.stage)
        write_json(self.rep/'formal_prediction_inputs.json',self.inputs)

    def load_external(self):
        p=self.out/'manifests/external_results.json'
        if not p.exists():
            self.integrity.append({'experiment':'EXTERNAL_RERUN','status':'MISSING','orders':2,'folds':10,'stages':4,'statistical_n':10});return
        ex=read_json(p);assert ex['pass'];name='EXTERNAL_RERUN';self.external_name=ex['method']+' (our rerun)';self.external_source=ex['provenance']
        for order in ['paper','reverse']:
            for f in range(1,11):
                ms=[]
                for st in range(1,5):
                    key=f'{order}/fold{f}/stage{st}';info=ex['stages'][key];fp=Path(info['path']);assert file_hash(fp)==info['sha256'];rr=read_json(fp)['rows']
                    seen=[TASKS.index(t) for t in ORDERS[order][:st]];expected={r['slide_id']:r for p in seen for r in self.man['cells'][f'{f}|{TASKS[p]}|test']['records']}
                    assert len(rr)==len(expected) and {r['slide_id'] for r in rr}==set(expected)
                    for r in rr:assert r['true_label']==expected[r['slide_id']]['global_label'] and r['patient_id']==expected[r['slide_id']]['patient_id']
                    met=metrics(rr,seen);ms.append(met)
                    self.stage.append({'experiment':name,'order':order,'fold':f,'stage':st,**{k:met[k] for k in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','TP']}})
                lg=longitudinal(ms,order);self.summary[name,order,f]=lg;self.last[name,order,f]=[rr]
                self.R.append({'experiment':name,'config':'official','fold':f,'seed':f,'order':order,'task_columns':ORDERS[order],'R':lg['R']})
                self.fold.append({'experiment':name,'fold':f,'order':order,'seeds_averaged':1,**{k:v for k,v in lg.items() if k!='R'}})
                self.perseed.append({'experiment':name,'config':'official','fold':f,'seed':f,'order':order,**{k:v for k,v in lg.items() if k!='R'}})
        self.external=True;self.integrity.append({'experiment':name,'orders':2,'folds':10,'stages':4,'status':'VALIDATED','statistical_n':10})

    def vals(self,name,order,metric):return [self.summary[name,order,f][metric] for f in range(1,11)]
    def summary_row(self,name,order):
        r={'experiment':name,'order':order,'n_folds':10}
        for k in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','TP','Forgetting','BWT','AIA']:
            v=self.vals(name,order,k);r.update({k:float(np.mean(v)),k+'_sd':float(np.std(v,ddof=1)),k+'_percent':100*float(np.mean(v)),k+'_sd_pp':100*float(np.std(v,ddof=1))})
        return r
    def cell(self,name,metric):
        return ' / '.join(f'{np.mean(self.vals(name,o,metric))*100:.2f} ± {np.std(self.vals(name,o,metric),ddof=1)*100:.2f}' for o in ['paper','reverse'])

    def bootstrap(self,names):
        draws={(name,o,k,'slide'):np.zeros(1000) for name in names for o in ['paper','reverse'] for k in FIELDS}
        draws.update({(name,o,k,'patient'):np.zeros(10000) for name in names for o in ['paper','reverse'] for k in FIELDS})
        tg=torch.Generator().manual_seed(0);rng=np.random.default_rng(20261014)
        for f in range(1,11):
            for p in range(4):
                records=self.man['cells'][f'{f}|{TASKS[p]}|test']['records'];ids=[r['slide_id'] for r in records]
                patients=sorted({r['patient_id'] for r in records});pidx={v:i for i,v in enumerate(patients)};pi=np.array([pidx[r['patient_id']] for r in records]);counts=np.bincount(pi,minlength=len(patients))
                si=torch.randint(len(ids),(1000,len(ids)),generator=tg).numpy();ci=rng.integers(0,len(patients),size=(10000,len(patients)));den=counts[ci].sum(1)
                for name in names:
                    for order in ['paper','reverse']:
                        runs=[{r['slide_id']:r for r in rr if r['true_task']==p} for rr in self.last[name,order,f]]
                        assert all(set(rr)==set(ids) for rr in runs)
                        for metric,col in FIELDS.items():
                            # Average seed correctness, not probabilities or ensemble decisions.
                            values=np.mean([[float(rr[s][col]==rr[s]['true_label']) for s in ids] for rr in runs],axis=0)
                            draws[name,order,metric,'slide']+=values[si].mean(1)/40
                            cs=np.bincount(pi,weights=values,minlength=len(patients))
                            draws[name,order,metric,'patient']+=cs[ci].sum(1)/den/40
            print('bootstrap fold',f,flush=True)
        return draws

    def paired(self):
        pairs=[('ROUTED_MEAN','LIN8_MEAN'),('B_TOP64_RIDGE','ROUTED_MEAN'),('B_TOP64_RIDGE','B_TOP64_COS'),('A_TOP64_RIDGE','A_TOP64_COS'),
               ('A_TOP64_COS','B_TOP64_COS'),('A_TOP64_RIDGE','B_TOP64_RIDGE'),('B_TOP64_FIXED','B_FOUR16_FIXED'),('A_TOP64_FIXED','A_FOUR16_FIXED'),
               ('A_TOP64_FIXED_3SEED','A_FOUR16_FIXED_3SEED'),('B_TOP64_FIXED','RANDOM64_RIDGE_FIXED'),('B_TOP64_FIXED','ROUTED_MEAN_FIXED'),('RANDOM64_RIDGE_FIXED','ROUTED_MEAN_FIXED')]
        if self.external:pairs.append(('B_TOP64_RIDGE','EXTERNAL_RERUN'))
        names=sorted({n for p in pairs for n in p});draws=self.bootstrap(names);rows=[]
        for a,b in pairs:
            for order in ['paper','reverse']:
                for metric in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','Forgetting','BWT','AIA']:
                    row={'first':a,'second':b,'direction':a+' minus '+b,'order':order,'metric':metric,'better_direction':'lower' if metric=='Forgetting' else 'higher',**sign(self.vals(a,order,metric),self.vals(b,order,metric),metric=='Forgetting')}
                    if metric in FIELDS:
                        for kind,n,seed in [('slide',1000,0),('patient',10000,20261014)]:
                            ci=100*np.quantile(draws[a,order,metric,kind]-draws[b,order,metric,kind],[.025,.975]);row.update({kind+'_ci_low_pp':float(ci[0]),kind+'_ci_high_pp':float(ci[1]),kind+'_resamples':n,kind+'_seed':seed})
                        row['CI_definition']='paired (fold,task)-stratified percentile; slide or patient-cluster; slides weighted within task; tasks/folds equal; shared draws across methods/seeds'
                    else:row['CI_definition']='no slide CI for longitudinal summary'
                    row['CI_scope']='test sampling conditional on fitted models; overlapping CV train sets; no full training uncertainty'
                    rows.append(row)
        for order in ['paper','reverse']:
            for metric in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','Forgetting','BWT','AIA']:
                subset=sorted([r for r in rows if r['order']==order and r['metric']==metric],key=lambda r:r['p_raw']);prev=0.
                for i,r in enumerate(subset):prev=max(prev,min(1.,(len(subset)-i)*r['p_raw']));r['p_holm']=prev;r['holm_family']=f'{order}/{metric}: {len(subset)} declared pairs'
        csv_write(self.rep/'paired_comparisons.csv',rows);self.pairs=rows
        for filename,names in [('selection_four16_vs_top64',['A_TOP64_FIXED','B_TOP64_FIXED','A_TOP64_FIXED_3SEED']),('selection_four16_vs_top64_5seed',['A_TOP64_FIXED','B_TOP64_FIXED']),('selection_four16_vs_top64_3seed',['A_TOP64_FIXED_3SEED','B_TOP64_FIXED'])]:
            csv_write(self.rep/(filename+'.csv'),[r for r in rows if r['first'] in names and 'FOUR16' in r['second']])
        return rows

    def tables(self):
        caption='Percent mean ± sample SD across 10 folds; original / reverse. A rows average five seeds within fold; external rerun uses seed=fold ID. For internal rows, Masked gives true task for both evidence and final class pair (oracle WP); external rerun Masked uses true-task pair argmax. Starred rows are literature reported results with original definitions; new rows from this study only.'
        lit=read_json(self.out/'manifests/literature_rows.json');table=[]
        for method in dict.fromkeys(r['method'] for r in lit):
            rr={r['order']:r for r in lit if r['method']==method};r=rr['paper'];row={'Method':method,'Buf.':r['buffer'],'BP':'yes' if r['BP'] else 'no','origin':'reported','citation':r['citation']}
            for title,col in [('ACC','ACC'),('Masked','Masked_ORACLE'),('Forgetting','Forgetting')]:row[title]=' / '.join('—' if rr[o][col] is None else f'{rr[o][col]*100:.2f} ± {rr[o][col+"_sd"]*100:.2f}' for o in ['paper','reverse'])
            table.append(row)
        new=[('LIN8_MEAN','Ridge, 8 classes, mean','no'),('B_TOP64_RIDGE','PathSelect (one-shot top-64)','no'),('A_TOP64_RIDGE','+ residual head','yes')]
        if self.external:new.insert(0,('EXTERNAL_RERUN',self.external_name,'yes'))
        else:table.append({'Method':'External official rerun','Buf.':0,'BP':'yes','ACC':'MISSING','Masked':'MISSING','Forgetting':'MISSING','origin':'required incomplete'})
        for name,label,bp in new:table.append({'Method':label,'Buf.':0,'BP':bp,'ACC':self.cell(name,'ACC'),'Masked':self.cell(name,'Masked_ORACLE'),'Forgetting':self.cell(name,'Forgetting'),'origin':'this study','citation':name})
        render_table(self.rep,'table1',table,['Method','Buf.','BP','ACC','Masked','Forgetting'],caption);atomic_bytes(self.rep/'table1_preview.md',(self.rep/'table1.md').read_bytes())
        ladder=[]
        for i,name in enumerate(['LIN8_MEAN','ROUTED_MEAN','B_TOP64_RIDGE']):
            for order in ['paper','reverse']:
                r=self.summary_row(name,order);r['row']='abc'[i];r['gamma']=self.exp['aliases'][name].split('_g')[-1]
                if i:
                    before=['LIN8_MEAN','ROUTED_MEAN'][i-1];paired=next(p for p in self.pairs if p['first']==name and p['second']==before and p['order']==order and p['metric']=='ACC')
                    r.update({k:v for k,v in paired.items() if k not in ['first','second','order']})
                ladder.append(r)
        render_table(self.rep,'table2_ladder',ladder,['row','experiment','order','gamma','ACC_percent','ACC_sd_pp','Masked_ORACLE_percent','Masked_ORACLE_sd_pp','mean_difference_pp','wins','losses','ties','p_raw','slide_ci_low_pp','slide_ci_high_pp','patient_ci_low_pp','patient_ci_high_pp'],caption+' Delta is current minus preceding row; pp.')
        fact=[]
        for name in ['B_TOP64_COS','B_TOP64_RIDGE','A_TOP64_COS','A_TOP64_RIDGE']:
            for order in ['paper','reverse']:fact.append({**self.summary_row(name,order),'gamma_tp':.001,'gamma_wp':float(self.alias[name].split('_g')[-1]) if name.endswith('RIDGE') else None,'n_seeds':5 if name.startswith('A') else 1,'selection_score':'residual head' if name.startswith('A') else 'text similarity','readout':'ridge' if name.endswith('RIDGE') else 'text cosine'})
        render_table(self.rep,'table2_factorial',fact,['selection_score','readout','order','gamma_wp','ACC_percent','ACC_sd_pp','Masked_ORACLE_percent','Masked_ORACLE_sd_pp'],caption+' Cosine rows retain the supervised router; they are not zero-shot pipelines.')
        csv_write(self.rep/'random64_control.csv',[self.summary_row(n,o) for n in ['B_TOP64_FIXED','RANDOM64_RIDGE_FIXED','ROUTED_MEAN_FIXED'] for o in ['paper','reverse']])
        csv_write(self.rep/'budget_sweep.csv',[{**self.summary_row('B_TOP64_FIXED' if k==64 else f'B_TOP{k}_FIXED',o),'K':k,'gamma_tp':.001,'gamma_wp':.001} for k in [16,32,64,128,256] for o in ['paper','reverse']]+[{**self.summary_row('ROUTED_MEAN_FIXED',o),'K':'all','gamma_tp':.001,'gamma_wp':.001} for o in ['paper','reverse']])
        csv_write(self.rep/'A_seed_subsets.csv',[self.summary_row(n,o) for n in ['A_TOP64_RIDGE','A_TOP64_RIDGE_3SEED','A_TOP64_COS','A_TOP64_COS_3SEED','A_TOP64_FIXED','A_TOP64_FIXED_3SEED','A_FOUR16_FIXED','A_FOUR16_FIXED_3SEED'] for o in ['paper','reverse']])
        csv_write(self.rep/'gamma_test_sensitivity.csv',[{**self.summary_row(f'B_TOP64_GAMMA_{g}',o),'gamma_wp':g,'status':'descriptive/post hoc; does not modify validation selection'} for g in [1e-5,1e-4,1e-3,1e-2,1e-1] for o in ['paper','reverse']])

    def per_task(self):
        per_task=[];per_class=[];conf=[];router=[];pooled=[]
        for order in ['paper','reverse']:
            allrows=[]
            for f in range(1,11):
                rr=self.last['B_TOP64_RIDGE',order,f][0];allrows+=rr
                for p in range(4):
                    r=[x for x in rr if x['true_task']==p];d={'order':order,'fold':f,'task':TASKS[p],'n':len(r),'TP':np.mean([x['pred_task']==p for x in r])}
                    d.update({k:float(np.mean([x[col]==x['true_label'] for x in r])) for k,col in FIELDS.items()});per_task.append(d)
                for c in range(8):
                    r=[x for x in rr if x['true_label']==c];per_class.append({'order':order,'fold':f,'class':c,'class_name':['ESAD','ESCC','CCRCC','PRCC','IDC','ILC','LUAD','LUSC'][c],'n':len(r),**{k:float(np.mean([x[col]==c for x in r])) for k,col in FIELDS.items()}})
            for true in range(8):
                for pred in range(8):conf.append({'order':order,'true_class':true,'predicted_class':pred,'count':sum(r['true_label']==true and r['pred_label']==pred for r in allrows),'aggregation':'pooled slide counts'})
            for true in range(4):
                for pred in range(4):router.append({'order':order,'true_task':TASKS[true],'predicted_task':TASKS[pred],'count':sum(r['true_task']==true and r['pred_task']==pred for r in allrows),'aggregation':'pooled slide counts'})
            pooled.append({'order':order,'n':len(allrows),'TP':np.mean([r['true_task']==r['pred_task'] for r in allrows]),**{k:np.mean([r[col]==r['true_label'] for r in allrows]) for k,col in FIELDS.items()},'aggregation':'pooled slides; not task/fold macro'})
        def summarize(raw,key,metrics):
            out=[]
            for order in ['paper','reverse']:
                for value in sorted({r[key] for r in raw}):
                    rows=[r for r in raw if r[key]==value and r['order']==order];assert len(rows)==10
                    out.append({'order':order,key:value,'n_folds':10,'n_slides':sum(r['n'] for r in rows),**{m:float(np.mean([r[m] for r in rows])) for m in metrics},**{m+'_sd':float(np.std([r[m] for r in rows],ddof=1)) for m in metrics}})
            return out
        csv_write(self.rep/'per_task_fold.csv',per_task);csv_write(self.rep/'per_class_fold.csv',per_class)
        csv_write(self.rep/'per_task.csv',summarize(per_task,'task',['TP']+list(FIELDS)));csv_write(self.rep/'per_class.csv',[{**r,'class_name':['ESAD','ESCC','CCRCC','PRCC','IDC','ILC','LUAD','LUSC'][r['class']]} for r in summarize(per_class,'class',list(FIELDS))])
        csv_write(self.rep/'confusion.csv',conf);csv_write(self.rep/'router_errors.csv',router);csv_write(self.rep/'esca_routing_errors.csv',[r for r in router if r['true_task']=='tcga_esca']);csv_write(self.rep/'pooled_counts.csv',pooled)

    def run(self):
        self.load();self.paired();self.tables();self.per_task();csv_write(self.rep/'completeness.csv',self.integrity)
        write_json(self.rep/'integrity_report.json',{'pass':self.external,'internal_predictions_validated':True,'external_complete':self.external,'source':self.host,'checks':self.integrity,'inputs':len(self.inputs),'reported_vs_new_separated':True})
        atomic_bytes(self.rep/'metrics_mapping.txt',b'ACC: stage4 task macro, then seed-within-fold then 10-fold mean/sample SD.\nMasked (paper): Masked_ORACLE = true-task evidence and true-task pair.\nLegacy: Masked_ROUTED_EVIDENCE = predicted-task evidence, true-task pair; not primary.\nLIN8/ROUTED_MEAN share subtype W and oracle outputs.\nForgetting: mean over first 3 tasks [max stages j..3 R[t,j] - R[4,j]], no clamp. BWT: R[4,j]-R[j,j].\nAIA: equal 4-stage mean. Two orders not independent. CV training sets overlap.\n')
        for name in ['per_seed_fold','stage_curves','per_task','per_class','per_task_fold','per_class_fold','pooled_counts']:
            path=self.rep/(name+'.csv');rows=list(csv.DictReader(path.open()))
            for row in rows:
                for key in ['ACC','Masked_ORACLE','Masked_ROUTED_EVIDENCE','TP','Forgetting','BWT','AIA']:
                    if key in row:row[key+'_percent']=100*float(row[key])
                    if key+'_sd' in row:row[key+'_sd_pp']=100*float(row[key+'_sd'])
            csv_write(path,rows)
        print('reports built; external complete',self.external,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();setup(4);Report(a.output.resolve()).run()
