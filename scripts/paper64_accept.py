"""Final physical-cell and shared-artifact acceptance gate, independent of Slurm state."""
import argparse,collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper64_run import *

def run(out):
    host=read_json(out/'environment/compute.json');data=read_json(out/'manifests/data.json');proto=file_hash(out/'protocol/protocol.json');text_hash=digest(data['text']);checks=[];counts=collections.Counter();headhash={}
    expected_metadata={'source_fingerprint':host['source_fingerprint'],'environment_hash':host['environment_hash'],'data_hash':data['data_hash'],'text_hash':text_hash,'protocol_hash':proto,'namespace':out.name}
    def provenance(m):
        for k,v in expected_metadata.items():assert m[k]==v,(k,m[k],v)
    references={}
    for p in sorted((out/'heads').glob('fold*_seed*/*.pt')):
        d=checked_load(p);meta=read_json(p.with_suffix('.pt.done.json'))['metadata'];cell=data['cells'][f"{meta['fold']}|{meta['task']}|train"];provenance(meta)
        assert meta['actual_epochs']==5 and len(d['history'])==5 and d['train_ids']==cell['slide_ids']
        assert all(h['n']==cell['n_slides'] and {r[0] for r in h['per_slide']}==set(cell['slide_ids']) for h in d['history'])
        headhash[meta['fold'],meta['seed_id'],TASKS.index(meta['task'])]=file_hash(p)
    assert len(headhash)==200
    for p in sorted((out/'vectors').rglob('*.pt')):
        d=checked_load(p);m=read_json(p.with_suffix('.pt.done.json'))['metadata'];cell=data['cells'][f"{m['fold']}|{TASKS[m['task']]}|{m['split']}"];provenance(m);references[str(p.relative_to(out))]=file_hash(p)
        assert d['ids']==cell['slide_ids'] and len(d['ids'])==cell['n_slides']
        assert d['labels'].tolist()==[r['global_label'] for r in cell['records']]
        assert d['patients']==[r['patient_id'] for r in cell['records']]
        if m['seed_id']:
            for q,h in d['head_checksums'].items():assert h==headhash[m['fold'],m['seed_id'],int(q)]
        counts['vectors']+=1
    assert counts['vectors']==720
    router_checks=[]
    for fold in range(1,11):
        train=[checked_load(out/path_vec(fold,0,'train',p)) for p in range(4)]
        for order in ORDERS:
            for stage in range(1,5):
                st=checked_load(out/'ridge/routers'/f'fold{fold:02d}_{order}_{stage}.pt');seen=st['tasks']
                X=aug(torch.cat([train[p]['vectors']['MEAN'][:,0] for p in seen]));y=torch.cat([torch.full((len(train[p]['ids']),),p) for p in seen])
                Ar=X.T@X;Br=torch.stack([X[y==p].sum(0) for p in seen],1);wr=torch.linalg.solve(Ar+.001*torch.eye(513,dtype=torch.float64),Br)
                torch.testing.assert_close(st['A'],Ar,atol=1e-8,rtol=1e-7);torch.testing.assert_close(st['B'],Br,atol=1e-8,rtol=1e-7);torch.testing.assert_close(st['W'],wr,atol=1e-6,rtol=1e-7)
                err=0.;ix=[seen.index(p) for p in sorted(seen)]
                for split in ['train','val','test']:
                    for p in seen:
                        V=aug((train[p] if split=='train' else checked_load(out/path_vec(fold,0,split,p)))['vectors']['MEAN'][:,0]);a,b=V@st['W'],V@wr
                        torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-7);assert torch.equal(a[:,ix].argmax(1),b[:,ix].argmax(1)), 'router batch-reference near tie'
                        err=max(err,float((a-b).abs().max()))
                router_checks.append({'fold':fold,'order':order,'stage':stage,'max_W_difference':float((st['W']-wr).abs().max()),'max_logit_difference':err,'exact_predictions':True})
    csv_write(out/'reports/router_batch_reference.csv',router_checks)
    for marker in sorted((out/'ridge').rglob('*.pt.done.json')):
        m=read_json(marker);provenance(m['metadata']);p=marker.with_suffix('').with_suffix('');h=file_hash(p);assert h==m['sha256'];references[str(p.relative_to(out))]=h
    expected=[json.loads(l) for l in (out/'protocol/expected_physical_cells.jsonl').read_text().splitlines()]
    for cell in expected:
        p=out/cell['prediction_path'];m=read_json(p.with_suffix('.pt.done.json'))['metadata'];d=checked_load(p);rr=d['rows'];f=cell['fold'];split=cell['split'];order=cell['order'];stage=cell['stage'];seen=[TASKS.index(t) for t in ORDERS[order][:stage]]
        for k,v in [('source_fingerprint',host['source_fingerprint']),('environment_hash',host['environment_hash']),('data_hash',data['data_hash']),('text_hash',text_hash),('protocol_hash',proto),('namespace',out.name)]:assert m[k]==v,(p,k)
        assert m['ridge_hash']==references[m['ridge_file']]
        assert m['evidence_hashes']==d['evidence_hashes']
        assert all(references[p]==h for p,h in zip(m['evidence_files'],m['evidence_hashes']))
        wanted={r['slide_id']:r for task in seen for r in data['cells'][f'{f}|{TASKS[task]}|{split}']['records']}
        assert len(rr)==len(wanted) and {r['slide_id'] for r in rr}==set(wanted)
        for r in rr:
            w=wanted[r['slide_id']];assert r['true_label']==w['global_label'] and r['patient_id']==w['patient_id'] and r['true_task'] in seen
        assert d['metrics']==metrics(rr,seen)
        counts[split]+=1;checks.append({'path':cell['prediction_path'],'sha256':file_hash(p),'rows':len(rr),'status':'VALIDATED'})
    assert counts['val']==3680 and counts['test']==4880
    assert len(list((out/'ridge/routers').glob('*.pt')))==80
    # Every pair of top/four and cosine/ridge uses the recorded shared heads/evidence.
    aliases=read_json(out/'protocol/expanded_manifest.json')['aliases'];maps=read_json(out/'protocol/table_map.json');cross=[]
    for table,name in maps.items():cross.append({'table_cell':table,'experiment_alias':name,'canonical_configuration':aliases[name],'reference_rule':'same prediction files; no independently computed table cell'})
    csv_write(out/'reports/cross_table_cells.csv',cross);csv_write(out/'reports/artifact_validation.csv',checks)
    assert read_json(out/'reports/integrity_report.json')['pass'];assert read_json(out/'reports/costs_integrity.json')['pass'];assert read_json(out/'reports/pilot_full_repeat.json')['pass']
    external=read_json(out/'manifests/external_results.json');assert external['pass'] and len(external['stages'])==80
    for info in external['stages'].values():assert file_hash(Path(info['path']))==info['sha256']
    mapping={'internal_heads':200,**counts,'external_stages':80,'folds':10,'orders':2,'stages':4,'test_slides':2835,'test_patients':2645}
    write_json(out/'reports/acceptance.json',{'pass':True,'counts':mapping,'source':host,'verified_at':now(),'primary':'B_TOP64_RIDGE','smoke_pilot_excluded':True,'training_source_snapshot_immutable':True,'source_checks':read_json(out/'reports/helper_source_parity.json')})
    print('ACCEPTED physical outputs',mapping,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();setup(4);run(a.output.resolve())
