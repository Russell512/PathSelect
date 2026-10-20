"""Scientific correctness checks independent of historical Mac result caches."""
import torch
import torch.nn.functional as F
import numpy as np
import pytest
from scripts.paper64_common import select_indices,random_indices,Readout,mean_norm,aug
from scripts.paper64_run import predict,pick,metrics
from selector.cil_ops import one_shot,four_round
from selector.i6_expert import I6Expert

@pytest.mark.parametrize('n',[1,38,64,65,131])
@pytest.mark.parametrize('ties',[False,True])
def test_selector_exact_original_and_unique(n,ties):
    g=torch.Generator().manual_seed(71);Z=torch.randn(n,512,generator=g);s=torch.zeros(n) if ties else torch.randn(n,generator=g)
    a=select_indices(Z,s,'top64');b=select_indices(Z,s,'four16')
    assert torch.equal(a,one_shot(s,64)) and torch.equal(b,four_round(Z,s,1.5,64,16))
    for ix in [a,b]:assert len(ix)==min(n,64) and len(ix.unique())==len(ix) and int(ix.min())>=0 and int(ix.max())<n

@pytest.mark.parametrize('k',[16,32,64,128,256])
def test_budget_uses_raw_mean_then_l2(k):
    g=torch.Generator().manual_seed(3);Z=torch.randn(302,512,generator=g)*torch.linspace(0.1,10,302)[:,None]
    ix=select_indices(Z,torch.randn(302,generator=g),'top64',k)
    assert len(ix)==k
    torch.testing.assert_close(mean_norm(Z,ix),F.normalize(Z[ix].mean(0),dim=0),atol=0,rtol=0)
    assert not torch.allclose(mean_norm(Z,ix),F.normalize(F.normalize(Z[ix],dim=1).mean(0),dim=0))

def test_random_subset_invariant_and_without_replacement():
    for n in [1,38,64,511]:
        a=random_indices(n,42,1,'TCGA-sample');b=random_indices(n,42,1,'TCGA-sample')
        assert torch.equal(a,b) and len(a)==min(64,n) and len(a.unique())==len(a)
    assert not torch.equal(random_indices(511,42,1,'a'),random_indices(511,43,1,'a'))
    assert not torch.equal(random_indices(511,42,1,'a'),random_indices(511,42,2,'a'))

def test_incremental_subtype_all_stages_gammas_and_class_columns():
    gen=torch.Generator().manual_seed(7);D={p:F.normalize(torch.randn(140,512,generator=gen),dim=1) for p in range(4)}
    ys={p:torch.arange(140)%2+2*p for p in range(4)}
    for order in [[0,1,2,3],[3,2,1,0]]:
        m=Readout();seen=[]
        for p in order:
            seen.append(p);m.add_task(p,D[p],ys[p]);X=aug(torch.cat([D[q] for q in seen]));y=torch.cat([ys[q] for q in seen])
            cls=sorted(y.unique().tolist());B=torch.stack([X[y==c].sum(0) for c in cls],1)
            torch.testing.assert_close(m.A,X.T@X,atol=1e-8,rtol=1e-7)
            for gamma in [1e-5,1e-4,1e-3,1e-2,1e-1]:
                m.gamma=gamma;s=m.state();ref=torch.linalg.solve(X.T@X+gamma*torch.eye(513,dtype=torch.float64),B)
                assert s['classes']==cls
                torch.testing.assert_close(s['W'],ref,atol=1e-8,rtol=1e-7)
                torch.testing.assert_close(X@s['W'],X@ref,atol=1e-8,rtol=1e-7)

def test_routed_evidence_has_no_true_task_input_and_oracle_is_separate():
    V=torch.zeros(4,4,512)
    for p in range(4):V[:,p,p]=1
    predicted=torch.tensor([3,2,1,0]);true=torch.arange(4)
    assert torch.equal(pick(V,predicted).argmax(1),predicted)
    assert torch.equal(pick(V,true).argmax(1),true)
    assert not torch.equal(pick(V,predicted),pick(V,true))

def test_readouts_share_evidence_mean_oracle_and_restrict_seen_columns():
    gen=torch.Generator().manual_seed(20);V=F.normalize(torch.randn(6,512,generator=gen),dim=1)
    texts=F.normalize(torch.randn(8,512,generator=gen),dim=1);task=torch.tensor([3,2,2,3,2,3]);state={'classes':[4,5,6,7],'W':torch.randn(513,4,generator=gen,dtype=torch.float64)}
    original=V.clone()
    for mode in ['COS','RDG']:
        pred,sc=predict(V,task,state,texts,mode)
        assert torch.equal(pred//2,task) and torch.equal(V,original)
        oracle,_=predict(V,task,state,texts,mode);assert torch.equal(pred,oracle)
    lin,sc=predict(V,task,state,texts,'RDG',True)
    assert set(lin.tolist())<=set(state['classes'])

def test_seed_before_i6_creation_reproducible():
    torch.manual_seed(42);a=I6Expert(2);torch.manual_seed(42);b=I6Expert(2)
    assert a.n_params()==1033 and all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())


def test_report_sign_exact_and_seedfold_sampling_unit():
    from scripts.paper64_report import sign
    x=sign([1.]*10,[0.]*10)
    assert x['wins']==10 and x['n_folds']==10 and x['p_raw']==2/1024
    y=sign([1.]*10,[1.]*10)
    assert y['ties']==10 and y['n_nonties']==0 and y['p_raw']==1
    z=sign([0.]*10,[1.]*10,lower=True)
    assert z['wins']==10 and z['mean_difference_pp']==-100

def test_report_forgetting_excludes_final_stage_without_clamp():
    from scripts.paper64_report import longitudinal
    order=[3,2,1,0];rows=[[.9],[.8,.6],[.7,.65,.8],[.95,.7,.85,.8]];stages=[]
    for v in rows:
        stages.append({'ACC':sum(v)/len(v),'Masked_ORACLE':1.,'Masked_ROUTED_EVIDENCE':1.,'TP':1.,'per_task':[{'task':p,'ACC':a} for p,a in zip(order,v)]})
    out=longitudinal(stages,'paper')
    assert out['Forgetting']==pytest.approx(-.05)
    assert out['BWT']==pytest.approx(.2/3)
    assert out['AIA']==pytest.approx(np.mean([sum(v)/len(v) for v in rows]))


def test_scheduler_timeout_is_explicit_not_a_false_failure_or_retry(monkeypatch):
    from scripts import paper64_ops as ops
    import subprocess
    def timeout(*a,**kw):raise subprocess.TimeoutExpired(a[0],kw['timeout'])
    monkeypatch.setattr(ops.subprocess,'run',timeout)
    r=ops.command(['sbatch','unit.sbatch'])
    assert r['returncode']==124 and r['timeout'] and 'ambiguous' in r['stderr']
    queue,acct,raw=ops.poll([{'job_id':'123'}])
    assert queue==acct=={} and len(raw)==1 and raw[0]['timeout']

def test_bootstrap_seed_average_patient_clusters_and_stratum_weights():
    from scripts.paper64_report import Report,FIELDS
    from scripts.paper64_common import TASKS
    report=Report.__new__(Report);report.man={'cells':{}};report.last={}
    for f in range(1,11):
        all_a=[[],[]];all_b=[]
        for p in range(4):
            records=[]
            for i in range(3):
                sid=f'{f}/{p}/{i}';pid=f'{f}/{p}/'+('a' if i<2 else 'b');records.append({'slide_id':sid,'patient_id':pid})
                base={'slide_id':sid,'patient_id':pid,'true_task':p,'true_label':2*p}
                for seed in range(2):all_a[seed].append({**base,**{col:2*p+seed for col in FIELDS.values()}})
                correct=f==1 and p==0 and i<2
                all_b.append({**base,**{col:2*p+int(not correct) for col in FIELDS.values()}})
            report.man['cells'][f'{f}|{TASKS[p]}|test']={'records':records}
        for order in ['paper','reverse']:
            report.last['A',order,f]=all_a;report.last['B',order,f]=[all_b]
    d=report.bootstrap(['A','B'])
    for order in ['paper','reverse']:
        for metric in FIELDS:
            np.testing.assert_allclose(d['A',order,metric,'slide'],.5,atol=1e-14)
            np.testing.assert_allclose(d['A',order,metric,'patient'],.5,atol=1e-14)
    ci=np.random.default_rng(20261014).integers(0,2,size=(10000,2))
    expected=np.array([2.,0.])[ci].sum(1)/np.array([2.,1.])[ci].sum(1)/40
    np.testing.assert_allclose(d['B','paper','ACC','patient'],expected,atol=0,rtol=0)
    si=torch.randint(3,(1000,3),generator=torch.Generator().manual_seed(0)).numpy()
    np.testing.assert_allclose(d['B','paper','ACC','slide'],np.array([1.,1.,0.])[si].mean(1)/40,atol=0,rtol=0)
    assert np.array_equal(d['B','paper','ACC','patient'],d['B','reverse','ACC','patient'])
