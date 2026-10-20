"""Persistent scheduler monitor with bounded infrastructure retries; bugs need an agent."""
import argparse,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from paper64_ops import *

_verified={}

def checked_hash(path):
    path=Path(path);st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
    if key not in _verified:_verified[key]=sha(path)
    return _verified[key]

def valid_receipt(j):
    p=j.get('receipt')
    if not p:return False,'no experiment receipt declared'
    p=Path(p)
    if not p.exists():return False,'expected receipt missing'
    try:
        m=read(p)
        if not m.get('pass'):return False,'receipt not passed'
        for name,h in m.get('artifacts',{}).items():
            if not Path(name).exists() or checked_hash(name)!=h:return False,'output checksum mismatch: '+name
        return True,'receipt and artifact checksums valid'
    except Exception as e:return False,str(e)

def run(out,interval):
    with locked(out/'slurm/monitor.lock',False):
        pid={'pid':os.getpid(),'started':now(),'interval_s':interval,'session':out.name,'log_path':str(out/'logs/monitor.log')}
        write(out/'slurm/monitor.pid.json',pid);sequence=0;terminal={}
        while True:
            write(out/'slurm/heartbeat.json',{'time':now(),'sequence':sequence,'pid':os.getpid(),'phase':'polling'})
            jobs=lines(out/'slurm/jobs.jsonl');queue,acct,raw=poll(jobs);states=[]
            active=[j['job_id'] for j in jobs if j['job_id'] in queue]
            for jid in active:
                result=command(['scontrol','show','job',jid,'-o'],timeout=10);raw.append(result)
                if result['returncode']:break
            write(out/'slurm/last_poll.json',raw)
            append(out/'slurm/poll_history.jsonl',{'at':now(),'active_ids':active,'scheduler':raw})
            for j in jobs:
                info=queue.get(j['job_id'],acct.get(j['job_id'],{'state':'ACCOUNTING_DELAY'}));state=info['state'];ok,why=valid_receipt(j)
                label='COMPLETE_VALIDATED' if ok else state
                if state=='COMPLETED' and not ok:label='NEEDS_AGENT'
                if state in {'FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT'} and not ok:label='NEEDS_AGENT'
                if state in {'NODE_FAIL','PREEMPTED'} and not ok and 'command' in j:
                    terminal.setdefault(j['job_id'],time.monotonic())
                    lineage=j.get('lineage',j['job_id']);prior=[r for r in jobs if r.get('lineage')==lineage]
                    if time.monotonic()-terminal[j['job_id']]>=180 and not any(r.get('retry_of')==j['job_id'] for r in jobs):
                        if len(prior)<2:
                            result=command(j['command']);append(out/'slurm/retries.jsonl',{'at':now(),'job_id':j['job_id'],'reason':state,'result':result})
                            if result['returncode']==0:
                                fresh={**j,'job_id':result['stdout'].strip().split(';')[0],'retry_of':j['job_id'],'lineage':lineage,'submitted':now()}
                                append(out/'slurm/jobs.jsonl',fresh);label='RETRIED'
                            else:label='NEEDS_AGENT'
                        else:label='NEEDS_AGENT'
                if j.get('superseded_by'):label='SUPERSEDED_BY_'+str(j['superseded_by'])
                progress={}
                root=Path(j.get('output',out))
                beats=list((root/'logs').glob('*.heartbeat.json'))
                matching=[read(p) for p in beats if read(p).get('job_id')==j['job_id']]
                if matching:progress=max(matching,key=lambda x:x['last_checked'])
                states.append({'job_id':j['job_id'],'phase':j.get('phase'),**info,'validation':why,'label':label,'progress':progress})
            sequence+=1
            delivery=read(out/'DELIVERY.json') if (out/'DELIVERY.json').exists() else {}
            status={'study':out.name,'heartbeat':now(),'sequence':sequence,'monitor':pid,'jobs':states,
                    'scheduler_reachable':all(x['returncode']==0 for x in raw),
                    'state':'COMPLETE' if delivery.get('accepted') else ('SCHEDULER_UNREACHABLE' if any(x['returncode'] for x in raw) else ('NEEDS_AGENT' if any(s['label']=='NEEDS_AGENT' for s in states) else 'IN_PROGRESS')),
                    'formal_complete':bool(delivery.get('accepted')),'automatic_code_repair':False}
            write(out/'status.json',status);write(out/'slurm/heartbeat.json',{'time':now(),'sequence':sequence,'pid':os.getpid()})
            print(now(),sequence,[(s['job_id'],s['label']) for s in states],flush=True)
            if delivery.get('accepted'):break
            time.sleep(interval)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--interval',type=int,default=90);x=a.parse_args();run(x.output.resolve(),x.interval)
