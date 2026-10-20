"""Generate manuscript update mappings and a self-contained accepted-study report."""
import argparse,csv,json,statistics,hashlib,datetime,io,os,tempfile
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def csvread(p):return list(csv.DictReader(Path(p).open()))
def atomic_text(p,text):
 p=Path(p);fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.',suffix='.tmp')
 with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def csvwrite(p,rows):
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows);atomic_text(p,f.getvalue())
def run(out):
 rep=out/'reports';accept=load(rep/'acceptance.json');assert accept['pass'];data=csvread(rep/'per_fold.csv');pairs=csvread(rep/'paired_comparisons.csv');selection=load(out/'protocol/selection.json');source=load(out/'environment/source.json');ex=load(out/'manifests/external_results.json')
 def values(name,order,metric):return [float(r[metric]) for r in data if r['experiment']==name and r['order']==order]
 def cell(name,metric='ACC'):
  return ' / '.join(f'{100*statistics.mean(values(name,o,metric)):.3f} ± {100*statistics.stdev(values(name,o,metric)):.3f}' for o in ['paper','reverse'])
 def pair(a,b):
  rr=[r for r in pairs if r['first']==a and r['second']==b and r['metric']=='ACC']
  return ' / '.join(f"{float(r['mean_difference_pp']):+.3f} pp; W/L/T={r['wins']}/{r['losses']}/{r['ties']}; p={float(r['p_raw']):.6g}; slide CI=[{float(r['slide_ci_low_pp']):+.3f},{float(r['slide_ci_high_pp']):+.3f}]; patient CI=[{float(r['patient_ci_low_pp']):+.3f},{float(r['patient_ci_high_pp']):+.3f}]" for o in ['paper','reverse'] for r in rr if r['order']==o)
 mapping=[]
 def add(old,target,new,action):mapping.append({'old_statement_or_value':old,'new_cell_metric':target,'new_value_original_reverse':new,'required_update':action})
 def numeric_rows(filename,keys):
  return [{k:r[k] for k in keys if k in r} for r in csvread(rep/filename)]
 add('四輪各選16 patches；lambda=1.5','B_TOP64_RIDGE selection','一次 one_shot(score, K=64)','方法段、主方法流程圖、pseudo-code與caption改成一次top-K；four16只保留固定gamma補充對照。')
 add('FINAL-A as main method','Table1_PathSelect','FINAL-B / B_TOP64_RIDGE','主方法固定無I6；A僅作residual-head消融。')
 add('Table2(b) task routing=random64','Table2_b','ROUTED_MEAN: whole-slide mean router + whole-slide mean subtype ridge','保留原whole-mean baseline；random64另列RANDOM64_RIDGE_FIXED。見table2_b_audit.md。')
 for old,name,metric in [('0.921 ± 0.022','B_TOP64_RIDGE','ACC'),('Masked 0.940 (legacy source must be distinguished)','B_TOP64_RIDGE','Masked_ORACLE'),('Forgetting paper 0.008 / reverse 0.023','B_TOP64_RIDGE','Forgetting'),('0.857','LIN8_MEAN','ACC'),('0.903','ROUTED_MEAN','ACC'),('residual+cosine 0.913','A_TOP64_COS','ACC'),('residual+ridge 0.925','A_TOP64_RIDGE','ACC')]:add(old,name+'/'+metric,cell(name,metric),'更新Abstract、Results、Table1/2與重複正文引用；不保留舊four16或Mac數值。')
 for old,a,b in [('+4.6 pp routing','ROUTED_MEAN','LIN8_MEAN'),('+1.8 pp text selection','B_TOP64_RIDGE','ROUTED_MEAN'),('+0.4 pp residual head','A_TOP64_RIDGE','B_TOP64_RIDGE'),('ridge vs text cosine','B_TOP64_RIDGE','B_TOP64_COS'),('residual ridge vs cosine','A_TOP64_RIDGE','A_TOP64_COS'),('external gain 3.8–5.3 pp and 9/10 wins','B_TOP64_RIDGE','EXTERNAL_RERUN')]:add(old,a+' minus '+b,pair(a,b),'以本批paired表更新增量、勝負平、raw p和兩種CI；不把不顯著稱等效。')
 add('three seeds 42–44 or seed repetitions counted as samples','A and random seeded rows','five seeds 42–46; mean within fold, n=10 folds; A three-seed supplement saved','五seed主表；三seed只作10/14對照；deterministic repeats不是新增樣本。')
 add('all ridge gamma=1e-3 or fixed old selected values','protocol/selection.json',json.dumps(selection['selected'],ensure_ascii=False),'逐representation使用validation全批選出的gamma；fixed selection/budget/random contrasts仍為1e-3。')
 add('Masked definitions combined','Masked_ORACLE vs Masked_ROUTED_EVIDENCE',cell('B_TOP64_RIDGE','Masked_ORACLE')+' ; legacy='+cell('B_TOP64_RIDGE','Masked_ROUTED_EVIDENCE'),'主表caption寫true task同時指定evidence和最後兩類；legacy另存audit。')
 add('cosine readout is zero-shot pipeline','Table2 factorial','supervised mean task router + text cosine readout','只能稱readout採text cosine；整個pipeline包含supervised router。')
 add('only processes 64 patches','timing.csv','all N patches read and scored; min(N,64) retained for equal pooling','方法、效率段和圖說避免只處理64 patches的說法；不含CONCH extraction。')
 metric_keys=['order','ACC_percent','ACC_sd_pp','Masked_ORACLE_percent','Masked_ORACLE_sd_pp']
 add('prior budget/gamma trend','budget_sweep.csv; gamma_test_sensitivity.csv',json.dumps({'budget':numeric_rows('budget_sweep.csv',['K']+metric_keys),'gamma_test_post_hoc':numeric_rows('gamma_test_sensitivity.csv',['gamma_wp']+metric_keys)},ensure_ascii=False),'K各自train/eval重建ridge、固定gamma；test gamma sweep標descriptive/post hoc，不回選主模型。')
 add('ESCA bottleneck / ILC performance','per_task.csv; per_class.csv; esca_routing_errors.csv',json.dumps({'task':numeric_rows('per_task.csv',['task','TP_percent','TP_sd_pp']+metric_keys),'class':numeric_rows('per_class.csv',['class_name']+metric_keys),'ESCA_pooled_counts':numeric_rows('esca_routing_errors.csv',['order','true_task','predicted_task','count','aggregation'])},ensure_ascii=False),'更新TP、oracle WP、CIL與ESCA錯分去向；pooled counts和fold/task macro分開。')
 add('Mac latency / storage numbers','timing_summary.csv; memory.csv; model_state_bytes.csv',json.dumps({'node':accept['source']['hostname'],'threads':4,'latency_seconds':numeric_rows('timing_summary.csv',['variant','phase','n','median_s','p95_s']),'state_bytes':numeric_rows('model_state_bytes.csv',['variant','head_bytes','readout_S_R_W_bytes','router_S_R_W_bytes','texts_bytes'])},ensure_ascii=False),'替換舊時間；共用process peak不可拆分給variant；模型tensor bytes與研究artifact disk storage分開。')
 add('unseen-data confirmatory claim','protocol/protocol.json','development-followup after prior test exploration','誠實註明舊test已看過；CI只涵蓋給定模型下test sampling，不涵蓋完整training uncertainty。')
 csvwrite(rep/'manuscript_number_map.csv',mapping)
 gs=[]
 for name,d in selection['selected'].items():
  for g,v in d['objectives'].items():gs.append({'representation':name,'gamma':g,'objective':v,'selected':float(g)==d['gamma'],'selected_boundary':d['boundary']})
 csvwrite(rep/'gamma_validation_summary.csv',gs)
 timing=csvread(rep/'timing_summary.csv');speed=[]
 for family in ['A','B']:
  for phase in ['t_selection_s','t_online_compute_s','t_read_compute_s']:
   a=next(r for r in timing if r['variant']==family+'64' and r['phase']==phase);b=next(r for r in timing if r['variant']==family+'four' and r['phase']==phase)
   speed.append({'family':family,'phase':phase,'top64_median_s':float(a['median_s']),'four16_median_s':float(b['median_s']),'speedup_four16_over_top64':float(b['median_s'])/float(a['median_s']),'top64_p95_s':float(a['p95_s']),'four16_p95_s':float(b['p95_s']),'node':accept['source']['hostname'],'threads':4,'read_compute_quantile_rule':'sum read+compute per observation before quantile'})
 csvwrite(rep/'selection_speedup.csv',speed)
 text=['PathSelect one-shot top-64 正式補實驗驗收報告','',f"主方法 B_TOP64_RIDGE；implementation commit {source['implementation_commit']}；source fingerprint {source['source_fingerprint']}。",'本批為開發後補實驗；所有自家新數字均來自本study。Original / reverse分開報告；百分比mean ± 十折sample SD（ddof=1）。','',
       'Table 2(b) 為whole-slide mean router + whole-slide mean subtype ridge，不是random64。12張N>64舊server向量與全片mean完全一致；公開Mac cache未提供，並未冒稱直接驗證其向量。完整來源見table2_b_audit.md。','',
       (rep/'table1_preview.md').read_text(),(rep/'table2_ladder.md').read_text(),(rep/'table2_factorial.md').read_text(),
       '固定gamma選片比較（delta=top64−four16）：',pair('B_TOP64_FIXED','B_FOUR16_FIXED'),pair('A_TOP64_FIXED','A_FOUR16_FIXED'),'',
       '隨機對照（五seeds先fold內平均）：',pair('B_TOP64_FIXED','RANDOM64_RIDGE_FIXED'),pair('B_TOP64_FIXED','ROUTED_MEAN_FIXED'),pair('RANDOM64_RIDGE_FIXED','ROUTED_MEAN_FIXED'),'',
       '完整性：'+json.dumps(accept['counts'],ensure_ascii=False),'主要Masked=oracle WP；legacy Masked另存。配對sign test以10 folds為單位，ties tolerance 1e-12；原始p和Holm補充均保存。Slide bootstrap 1000次torch seed0；patient cluster bootstrap 10000次seed20261014，均在(fold,task)內配對。兩orders與50 seed-fold不視為獨立樣本。','',
       'K sweep、gamma validation與post hoc test sensitivity、per-task/class、confusion與ESCA routing errors分別見同名CSV。所有方法段/圖說/正文數字更新項目見manuscript_number_map.csv；沒有修改paper/。','',
       '外部官方重跑：'+ex['method']+'；source '+ex['provenance']['commit']+'；20 runs / 80 stages；seed=fold ID；官方validation、scheduler、early stopping與checkpoint流程保留；12 epochs/task為上限。逐run實際epochs、機器、參數、原始predictions與checkpoint hashes見manifests/external_results.json。','',
       '計時使用相同physical node、CPU backend、4 threads，逐slide交錯A/B與top64/four16並丟棄暖機；實際從raw patch features完成router、score、selection、pooling與readout。全部patches仍被讀取和評分。read+compute先相加再取median/P95。共用peak RSS與Slurm MaxRSS按process/step標示，未拆成variant peak。','',
       '所有失敗、修復和重送保留於slurm/repairs.jsonl與jobs.jsonl；不選最佳retry。Pilot、smoke、repeat與歷史結果均未混入正式品質數字。','',f'產物根目錄：{out}',f'外部產物：{ex["provenance"]["checkout"]}/study','重跑與resume指令見REPRODUCE.txt；交接見HANDOFF.txt。']
 atomic_text(rep/'FINAL_REPORT.md','\n'.join(text)+'\n')
 atomic_text(out/'HANDOFF.txt','Study accepted only after reports/acceptance.json passes and DELIVERY.json is sealed.\nPrimary: B_TOP64_RIDGE; A is residual-head ablation.\nSource: '+str(out/'source'/source['source_fingerprint'][:16])+'\nAnalysis source may have a later separate commit; do not replace frozen training source.\nSource/data/environment/protocol hashes are checked before cache reuse.\nDo not reset the repository, overwrite previous outputs, modify datasets/paper/, cancel unrelated jobs, or push/merge.\nMonitor cannot repair arbitrary new Python failures; NEEDS_AGENT requires agent diagnosis.\nRead FINAL_REPORT, completeness, acceptance, Slurm ledger and REPRODUCE before resuming.\n')
 print('manuscript map and final report generated',len(mapping),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve())
