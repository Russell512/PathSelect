"""Lightweight scheduling utilities (stdlib only, safe on login node)."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
ACTIVE={'PENDING','RUNNING','REQUEUED','REQUEUE_FED','REQUEUE_HOLD','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','STAGE_OUT'}


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    with os.fdopen(fd,'w') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def append(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a') as f:f.write(json.dumps(data,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []
@contextlib.contextmanager
def locked(path,blocking=True):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB));yield
def command(cmd,timeout=45):
    try:
        r=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout)
        return {'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'checked':now()}
    except subprocess.TimeoutExpired as e:
        stdout=e.stdout.decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or '')
        return {'command':cmd,'returncode':124,'stdout':stdout,'stderr':f'scheduler command timed out after {timeout}s; submission acknowledgement may be ambiguous','checked':now(),'timeout':True}



def poll(jobs):
    ids=list(dict.fromkeys(j['job_id'] for j in jobs))
    if not ids:return {},{},[]
    q=command(['squeue','-h','-j',','.join(ids),'-o','%i|%T|%R|%N'],timeout=15)
    if q['returncode']:return {},{},[q]
    a=command(['sacct','-n','-P','-j',','.join(ids),'--format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS,NodeList,Reason'],timeout=15)
    queue={}
    for ln in q['stdout'].splitlines():
        v=ln.split('|')
        if len(v)>=4:queue[v[0]]={'state':v[1],'reason':v[2],'nodes':v[3]}
    acct={}
    for ln in a['stdout'].splitlines():
        v=ln.split('|')
        if len(v)>=7:acct[v[0]]={'state':v[1].split()[0].rstrip('+'),'exit_code':v[2],'elapsed':v[3],'max_rss':v[4],'nodes':v[5],'reason':v[6]}
    return queue,acct,[q,a]


def unit_valid(out,fold,seed,full=False):
    out=Path(out);p=out/'metrics'/f'fold{fold:02d}_seed{seed}.done.json'
    if not p.exists():return False,'no completion marker'
    try:
        m=read(p)
        if m['stage_keys']!=32 or m['heads']!=4:return False,'wrong counts'
        if len(m['artifacts'])!=56:return False,f"artifact count {len(m['artifacts'])} !=56"
        for name,h in m['artifacts'].items():
            ap=out/name;mp=ap.with_suffix(ap.suffix+'.done.json')
            if not ap.exists() or not mp.exists():return False,'missing artifact'
            am=read(mp)
            if am['sha256']!=h:return False,'marker checksum mismatch'
            for k in ['source_fingerprint','data_hash','environment_hash','protocol_hash']:
                if am['metadata'][k]!=m['metadata'][k]:return False,'provenance mismatch'
            if full and sha(ap)!=h:return False,'file checksum mismatch'
        return True,'validated checksums' if full else 'metadata and existence checked'
    except Exception as e:return False,str(e)
