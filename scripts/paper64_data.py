"""Run one complete (fold, seed/repeat) on an allocated CPU backend."""
from __future__ import annotations
import argparse
import resource
import traceback
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper64_common import *
from selector.cil_ops import train_selector
from selector.evaluate import slide_dataset, read_slide
from selector.incremental_ridge import IncrementalRidge
from scripts.preflight_paper64 import preflight


def stage_metrics(rows, seen):
    per=[]
    for p in seen:
        rr=[r for r in rows if r['true_task']==p]
        if not rr: raise ValueError('empty task evaluation')
        classes=[]
        for c in [2*p,2*p+1]:
            cc=[r for r in rr if r['true_label']==c]
            if not cc: raise ValueError('empty class evaluation')
            classes.append(sum(r['pred_label']==c for r in cc)/len(cc))
        per.append({'task':p,'n':len(rr),'cil':sum(r['pred_label']==r['true_label'] for r in rr)/len(rr),
                    'tp':sum(r['pred_task']==p for r in rr)/len(rr),
                    'oracle_wp':sum(r['oracle_pred']==r['true_label'] for r in rr)/len(rr),
                    'balanced_accuracy':sum(classes)/2,'class_accuracy':classes})
    return {'cil':sum(r['cil'] for r in per)/len(per),'tp':sum(r['tp'] for r in per)/len(per),
            'oracle_wp':sum(r['oracle_wp'] for r in per)/len(per),
            'balanced_accuracy':sum(r['balanced_accuracy'] for r in per)/len(per),'per_task':per,'n':len(rows)}


class DataRun:
    def __init__(self,out,fold,seed,smoke=False):
        if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('compute work requires a Slurm allocation')
        self.out,self.fold,self.seed,self.smoke=out,fold,seed,smoke
        self.cfg,self.exp=config()
        if not smoke and (fold not in self.exp['folds'] or seed not in [0]+self.exp['seeds']): raise ValueError('invalid formal unit')
        if torch.get_num_threads()>int(os.environ['SLURM_CPUS_PER_TASK']): raise ValueError('threads exceed allocation')
        for d in ['protocol','environment','manifests','heads','vectors','ridge','predictions','metrics','logs','slurm','reports']:
            (out/d).mkdir(parents=True,exist_ok=True)
        self.unit=f'fold{fold:02d}_seed{seed}'
        self.manifest=read_json(out/'manifests/data.json')
        pf=read_json(out/'manifests/preflight.json')
        if not pf['pass']: raise ValueError('preflight blocked')
        self.src=source_identity()
        self.env=environment()
        self.base={'base_commit':self.exp['base_commit'],'implementation_commit':self.src['implementation_commit'],
                   'source_fingerprint':self.src['source_fingerprint'],'environment_hash':digest(self.env),
                   'data_hash':self.manifest['data_hash'],'text_hash':digest(self.manifest['text']),
                   'protocol_hash':file_hash(out/'protocol/protocol.json'),'device':'cpu','feature_dtype':'float32',
                   'ridge_dtype':'float64','threads':self.cfg['threads'],'fold':fold,'namespace':out.name,
                   'pooling':self.exp['pooling'],'training':{k:self.exp[k] for k in ['rank','epochs','lr','weight_decay']}}
        self.own={**self.base,'seed_id':seed,'repeat_id':seed}
        self.runtime={**self.env,'hostname':socket.gethostname(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
                      'started':now(),'source':self.src,'fold':fold,'seed_id':seed,'loadavg':list(os.getloadavg())}
        with lock(out/'environment/.host.lock'):
            hp=out/'environment/compute.json'
            signature={'hostname':socket.gethostname(),'environment_hash':digest(self.env),'source_fingerprint':self.src['source_fingerprint']}
            if hp.exists() and read_json(hp)!=signature: raise ValueError('mixed physical host/environment/source prohibited')
            write_json(hp,signature)
        write_json(out/'environment'/f'{self.unit}_{os.environ["SLURM_JOB_ID"]}.json',self.runtime)
        self.text=[torch.load(ROOT/self.manifest['text'][t]['path'],map_location='cpu',weights_only=True) for t in TASKS]
        for p,t in enumerate(TASKS):
            if file_hash(ROOT/self.manifest['text'][t]['path'])!=self.manifest['text'][t]['sha256']: raise ValueError('text changed')
        self.heads={};self.head_hash={};self.artifacts={};self.lastbeat=0

    def beat(self,phase,force=False,**kw):
        if force or time.monotonic()-self.lastbeat>=30:
            d={'last_checked':now(),'unit':self.unit,'phase':phase,'job_id':os.environ['SLURM_JOB_ID'],
               'hostname':socket.gethostname(),'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**kw}
            write_json(self.out/'logs'/f'{self.unit}.heartbeat.json',d)
            print(json.dumps(d),flush=True);self.lastbeat=time.monotonic()

    def dataset(self,p,split):
        ds,shift=slide_dataset({**self.cfg,'fold':self.fold},TASKS[p],p,split)
        cell=self.manifest['cells'][f'{self.fold}|{TASKS[p]}|{split}']
        if list(ds.sids)!=cell['slide_ids']: raise ValueError('dataset ID/order drift')
        if {str(s):int(ds.sid2label[s]) for s in ds.sids}!={r['slide_id']:r['local_label'] for r in cell['records']}:
            raise ValueError('label drift')
        return ds,shift

    def read(self,ds,shift,i):
        sid=ds.sids[i];task=TASKS[shift//2]
        fp=Path(ds.read_path)/(sid+'.'+ds.read_format)
        fmeta=self.manifest['features'][f'{task}/{sid}'];st=fp.stat()
        if st.st_size!=fmeta['bytes'] or st.st_mtime_ns!=fmeta['mtime_ns']: raise ValueError('feature file changed')
        t=time.perf_counter();r=read_slide(ds,shift,i);tr=time.perf_counter()-t
        feature_check(r.Z,r.sid)
        self.beat('read',task=task,slide=i,slide_id=sid)
        return r,tr

    def artifact(self,path,metadata,build,validate):
        path=self.out/path
        with lock(path.with_suffix(path.suffix+'.lock')):
            value=load_artifact(path,metadata)
            if value is None:
                value=build();validate(value);save_artifact(path,value,metadata)
            validate(value)
            marker=read_json(path.with_suffix(path.suffix+'.done.json'))
            self.artifacts[str(path.relative_to(self.out))]=marker['sha256']
        return value

    def train_head(self,p):
        task=TASKS[p];ds,shift=self.dataset(p,'train')
        n=min(2,len(ds)) if self.smoke else len(ds)
        epochs=1 if self.smoke else self.exp['epochs']
        metadata={**self.own,'task':task,'kind':'head','train_ids_hash':ids_hash(ds.sids[:n]),'actual_epochs':epochs}
        path=Path('heads')/self.unit/(task+'.pt')
        def build():
            random.seed(self.seed);np.random.seed(self.seed);torch.manual_seed(self.seed)
            head=I6Expert(2);assert head.n_params()==1033
            count=0
            def slides(g):
                nonlocal count
                for i in torch.randperm(n,generator=g).tolist():
                    r,tr=self.read(ds,shift,i)
                    self.beat('head_training',task=task,epoch=count//n+1,slide=count%n,n=n)
                    count+=1
                    yield tr,r.sid,r.Z,r.label-shift
            start=time.perf_counter()
            head,history=train_selector(slides,self.text[p]['f_txt'],self.text[p]['logit_scale'],epochs=epochs,
                  lr=self.exp['lr'],weight_decay=self.exp['weight_decay'],seed=self.seed,model=head,
                  log=lambda s:self.beat('head_training',force=True,task=task,message=s))
            return {'state_dict':head.state_dict(),'history':history,'wall_s':time.perf_counter()-start,
                    'train_ids':list(ds.sids[:n]),'n_params':head.n_params()}
        def valid(v):
            assert v['n_params']==1033 and v['train_ids']==list(ds.sids[:n]) and len(v['history'])==epochs
            assert all(e['n']==n for e in v['history'])
        value=self.artifact(path,metadata,build,valid)
        head=I6Expert(2);head.load_state_dict(value['state_dict']);head.eval();head.requires_grad_(False)
        self.heads[p]=head;self.head_hash[p]=file_hash(self.out/path)
        write_json(self.out/'heads'/self.unit/(task+'_history.json'),{k:v for k,v in value.items() if k!='state_dict'})
