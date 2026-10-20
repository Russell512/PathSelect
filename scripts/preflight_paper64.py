"""Audit original splits and features without creating replacement data."""
from __future__ import annotations
import argparse
import collections
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper64_common import *
from selector.evaluate import slide_dataset, read_slide
from selector.text_encoder import class_prompt_ensemble


def preflight(out, sample=False):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('full integrity scan requires compute node')
    cfg, exp = config()
    freeze_protocol(out, cfg, exp)
    root = cfg['dataset_root_dir']
    reference = read_json(ROOT / 'configs/benchmark_hashes.json')
    cells, files, feature_files, texts, samples = {}, {}, {}, {}, {}
    issues = []
    test_slides, test_patients = [], []
    for p, task in enumerate(TASKS):
        table = Path(root + cfg['path_table'].format(task, task.upper()))
        df = pd.read_csv(table, dtype={'patient_id':str, 'pathology_id':str})
        files[f'{task}/table'] = {'path': str(table), 'sha256': file_hash(table)}
        assert file_hash(table) == reference['files'][f'{task}/table'], 'table differs from benchmark'
        cp = ROOT / cfg['f_txt_cache_dir'] / f'f_txt_{task}.pt'
        assert file_hash(cp) == reference['text'][task]['sha256'], 'text cache differs from benchmark'
        ct = torch.load(cp, map_location='cpu', weights_only=True)
        names, _ = class_prompt_ensemble(task, cfg['class_prompt_path'])
        assert ct['class_names'] == names and ct['f_txt'].shape == (2,512)
        assert ct['f_txt'].dtype == torch.float32 and float(ct['logit_scale']) > 0
        finite_tree(ct)
        texts[task] = {'path':str(cp.relative_to(ROOT)), 'sha256':file_hash(cp), 'class_names':names,
                       'shape':list(ct['f_txt'].shape), 'dtype':str(ct['f_txt'].dtype), 'logit_scale':float(ct['logit_scale'])}
        for label, name in enumerate(names):
            assert set(df.loc[df.label == label, 'subtype']) == {name}
        task_tests, task_testp = [], []
        for fold in range(1, 11):
            sp = Path(root + cfg['path_split'].format(task, fold))
            files[f'{task}/fold_{fold}.npz'] = {'path':str(sp), 'sha256':file_hash(sp)}
            assert file_hash(sp) == reference['files'][f'{task}/fold_{fold}.npz'], 'split differs from benchmark'
            z = np.load(sp, allow_pickle=True)
            pts = {s:list(map(str, z[s+'_patients'])) for s in SPLITS}
            for a,b in [('train','val'),('train','test'),('val','test')]:
                assert not set(pts[a]) & set(pts[b]), (fold, task, a, b, 'patient leakage')
            for split in SPLITS:
                key = f'{fold}|{task}|{split}'
                ids = pts[split]
                assert len(ids) == len(set(ids)), (key, 'duplicate patients')
                absent = set(ids) - set(df.patient_id)
                assert not absent, (key, 'missing table patients', sorted(absent))
                sub = df[df.patient_id.isin(ids)]
                assert not sub.pathology_id.duplicated().any(), (key, 'duplicate slides')
                assert set(sub.label) <= {0,1}, (key, 'wrong local labels')
                assert all(sid[:12] == pid for sid,pid in zip(sub.pathology_id, sub.patient_id)), (key,'barcode mapping')
                ds, shift = slide_dataset({**cfg,'fold':fold},task,p,split)
                sids = list(map(str,ds.sids))
                assert len(sids) == len(sub) and set(sids) == set(sub.pathology_id), (key,'loader dropped rows')
                assert ids_hash(sids) == reference['cells'][key]['slide_sha256'], (key,'ext1 hash mismatch')
                refp = reference['cells'][key]
                assert ids_hash(ids) == refp['patient_sha256'] and len(ids) == refp['n_patients'], (key,'patient split mismatch')
                records = [{'slide_id':sid, 'patient_id':ds.sid2pid[sid], 'local_label':int(ds.sid2label[sid]),
                            'global_label':int(ds.sid2label[sid])+shift} for sid in sids]
                cells[key] = {'patient_ids':sorted(ids), 'slide_ids':sids, 'records':records,
                              'n_patients':len(ids), 'n_slides':len(sids), 'patient_sha256':ids_hash(ids),
                              'slide_sha256':ids_hash(sids), 'ext1_match':True}
                for sid in sids:
                    fkey = f'{task}/{sid}'
                    if fkey not in feature_files:
                        fp = Path(ds.read_path) / (sid+'.'+ds.read_format)
                        if not fp.is_file():
                            issues.append({'type':'missing_feature','task':task,'slide_id':sid,'path':str(fp)})
                        else:
                            st = fp.stat()
                            feature_files[fkey] = {'path':str(fp),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
                if sample and fold == 1:
                    for i in sorted(set([0,len(ds)//2,len(ds)-1])):
                        sid=ds.sids[i]; fp=Path(ds.read_path)/(sid+'.'+ds.read_format)
                        if not fp.exists(): continue
                        raw=torch.load(fp,map_location='cpu',weights_only=True)
                        rec=read_slide(ds,shift,i)
                        feature_check(rec.Z,rec.sid)
                        samples[f'{task}/{sid}']={'sha256':file_hash(fp),'shape':list(raw.shape),
                                                  'stored_dtype':str(raw.dtype),'loaded_dtype':str(rec.Z.dtype),'finite':True}
                if split=='test':
                    task_tests.extend(sids); task_testp.extend(ids)
        assert len(task_tests)==len(set(task_tests)), (task,'cross-fold test slide repeats')
        assert len(task_testp)==len(set(task_testp)), (task,'cross-fold test patient repeats')
        universe=set(cells[f'1|{task}|train']['slide_ids'])|set(cells[f'1|{task}|val']['slide_ids'])|set(cells[f'1|{task}|test']['slide_ids'])
        assert set(task_tests)==universe, (task,'test coverage')
        test_slides.extend(task_tests); test_patients.extend(task_testp)
    assert len(test_slides)==2835 and len(set(test_patients))==2645
    assert len(cells)==120
    for task,expected_n in zip(TASKS,[150,768,952,965]):
        assert sum(cells[f'{f}|{task}|test']['n_slides'] for f in range(1,11))==expected_n
    for i,(key,meta) in enumerate(feature_files.items()):
        raw=torch.load(meta['path'],map_location='cpu',weights_only=True)
        assert raw.ndim==2, (key,raw.shape)
        feature_check(raw.float(),key)
        meta.update(sha256=file_hash(meta['path']),shape=list(raw.shape),stored_dtype=str(raw.dtype),finite=True)
        if i%200==0: print(f'feature audit {i}/{len(feature_files)}',flush=True)
    manifest={'cells':cells,'files':files,'features':feature_files,'text':texts,'dataset_root':root,
              'reference_hashes':{'benchmark_hashes.json':file_hash(ROOT/'configs/benchmark_hashes.json')},
              'test_slides':len(test_slides),'test_patients':len(set(test_patients))}
    manifest['data_hash']=digest(manifest)
    with lock(out/'manifests/.lock'):
        path=out/'manifests/data.json'
        if path.exists() and read_json(path)!=manifest: raise ValueError('dataset manifest changed')
        write_json(path,manifest)
        if sample: write_json(out/'manifests/feature_samples.json',samples)
        write_json(out/'manifests/preflight.json',{'checked':now(),'pass':not issues,'sampled':sample,
                   'cells':len(cells),'feature_files':len(feature_files),'issues':issues,'data_hash':manifest['data_hash']})
    print(f'preflight cells={len(cells)} features={len(feature_files)} test={len(test_slides)} patients={len(set(test_patients))} issues={len(issues)} sampled={sample}',flush=True)
    if issues: raise ValueError(f'{len(issues)} missing feature files; see manifests/preflight.json')
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--sample',action='store_true')
    args=ap.parse_args();cfg,_=config();setup(cfg['threads']);preflight(args.output.resolve(),args.sample)
