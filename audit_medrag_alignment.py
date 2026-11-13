"""Read-only audit of both MedRAG projects using frozen inference artifacts.

Writes only audit outputs beside this script. Does not import/run experiment mains.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sys
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / '2026-2027-hold-submit-Electronics-Med-RAG'
SOURCE = TARGET / 'back-up'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE / '02_code'))
from data_prep import load_clean_corpus
from run_review_experiments import graph_rankings, rank_metrics, summarize, paired_comparison
from run_reviewer_revision_experiments import main_split, normalize_question, label_bonus_rankings


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()


def main():
    a = SOURCE / '02_code' / 'artifacts'
    output = HERE / 'audit_medrag'
    output.mkdir(exist_ok=True)
    checks = []
    def check(label, ok, detail=''):
        checks.append({'check': label, 'pass': bool(ok), 'detail': str(detail)})
    corpus = pd.read_parquet(a / 'corpus.parquet')
    fresh = load_clean_corpus(max_docs=-1)
    check('cached corpus equals cleaned raw dataset', corpus.equals(fresh))
    check('manuscripts identical', sha(HERE/'MDPI_template_ACS/Med-rag-highlight.tex') == sha(TARGET/'MDPI_template_ACS/Med-rag-highlight.tex'))
    for name in ['corpus.parquet', 'doc_embeddings.npy', 'entity_embeddings.npy', 'entity_names.json', 'graph.pkl']:
        check('shared artifact '+name, sha(a/name) == sha(HERE/'02_code/artifacts'/name))
    check('raw datasets identical', sha(HERE/'01_data/medquad.csv') == sha(SOURCE/'01_data/medquad.csv'))
    dev, test = main_split(corpus)
    ids = pd.concat([dev, test]).doc_id.astype(int).tolist()
    check('cached query order', json.loads((a/'review_all_query_doc_ids.json').read_text()) == ids)
    check('split sizes', (len(dev), len(test)) == (3349, 12996))
    check('disjoint document IDs', not set(dev.doc_id) & set(test.doc_id))
    vec = np.load(a/'review_all_query_embeddings.npy')[len(dev):]
    emb = np.load(a/'doc_embeddings.npy')
    ent = np.load(a/'entity_embeddings.npy')
    names = json.loads((a/'entity_names.json').read_text())
    with (a/'graph.pkl').open('rb') as f:
        graph = pickle.load(f)
    check('graph statistics', (len(graph), graph.number_of_edges(), len({v['community'] for _,v in graph.nodes(data=True)})) == (5125,19848,63))
    check('graph has no isolated nodes', all(d > 0 for _,d in graph.degree()))
    check('normalized embeddings', np.allclose(np.linalg.norm(emb,axis=1),1,atol=1e-5) and np.allclose(np.linalg.norm(vec,axis=1),1,atol=1e-5))
    r = SimpleNamespace(corpus=corpus.set_index('doc_id',drop=False), graph=graph,
                        doc_embeddings=emb, entity_embeddings=ent, entity_names=names,
                        doc_id_to_pos={int(d):i for i,d in enumerate(corpus.doc_id)})
    r._link_entities = lambda q,h: [names[j] for j in np.argsort(ent@q)[::-1][:h]]
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    scores,pos = index.search(vec.astype(np.float32),20)
    candidates = corpus.doc_id.to_numpy()[pos].astype(int)
    rankings = {'dense': candidates.tolist()}
    print('Recomputing entity and neighbor rankings',flush=True)
    rankings['entity'] = graph_rankings(r,vec,rankings['dense'],2,False)
    rankings['neighbors'] = graph_rankings(r,vec,rankings['dense'],2,True)
    rankings['label_bonus'] = label_bonus_rankings(r,vec,rankings['dense'],scores,0.05)
    ce = np.load(a/'review_cross_encoder_scores_len256.npy')
    check('complete cross-encoder scores', ce.shape == candidates.shape and np.isfinite(ce).all())
    rankings['cross_encoder'] = [[int(row[j]) for j in sorted(range(20),key=lambda j:(-float(v[j]),int(row[j])))] for row,v in zip(candidates,ce)]
    metrics = {k:rank_metrics(v,test.doc_id.to_numpy()) for k,v in rankings.items()}
    summaries = pd.DataFrame([summarize(k,v) for k,v in metrics.items()])
    expected = {'dense':(68.88,80.08),'entity':(69.44,80.45),'neighbors':(69.18,80.32), 'label_bonus':(69.64,80.74),'cross_encoder':(71.51,82.10)}
    for row in summaries.itertuples():
        check('manuscript primary result '+row.method, (round(row.hit_at_1_pct,2),round(row.mrr_at_20_pct,2)) == expected[row.method])
    summaries.to_csv(output/'recomputed-heldout.csv',index=False)
    pairs = []
    for base,method in [('dense','entity'),('dense','neighbors'),('entity','neighbors'),('dense','label_bonus'),('entity','label_bonus'),('dense','cross_encoder'),('entity','cross_encoder')]:
        pairs.append(paired_comparison(base,method,metrics[base],metrics[method]))
    pd.DataFrame(pairs).to_csv(output/'recomputed-paired.csv',index=False)
    check('neighbor direct wins/losses', (pairs[2]['top1_wins'],pairs[2]['top1_losses']) == (97,131))
    changes = sum(a[0]!=b[0] for a,b in zip(rankings['dense'],rankings['neighbors']))
    check('104 changed top documents',changes == 104)
    counts = corpus.question.map(normalize_question).value_counts()
    unique = test.question.map(normalize_question).map(counts).eq(1).to_numpy()
    check('10239 unique heldout questions',unique.sum()==10239)
    subset = pd.DataFrame([summarize(k,{m:v[unique] for m,v in values.items()}) for k,values in metrics.items()])
    subset.to_csv(output/'recomputed-unambiguous.csv',index=False)
    # Compare all five subgroup summaries with the frozen results, not just rounded manuscript values.
    stored = pd.read_csv(SOURCE/'03_result/result-reviewer-unambiguous-all-methods.csv')
    check('unique-question metrics match stored CSV', np.allclose(subset.select_dtypes('number'),stored.select_dtypes('number'),atol=1e-9))
    raw = pd.read_csv(SOURCE/'01_data/medquad.csv')
    raw_counts = corpus.question.value_counts()
    overlap = set(dev.question.map(normalize_question)) & set(test.question.map(normalize_question))
    diagnostics = {'raw_rows':len(raw),'clean_rows':len(corpus),'labels':corpus.focus_area.nunique(),
                   'repeated_raw_strings':int((raw_counts>1).sum()),'records_in_repeated_raw_strings':int(raw_counts[raw_counts>1].sum()),
                   'shared_normalized_strings':len(overlap),'test_records_with_exact_dev_overlap':int(test.question.map(normalize_question).isin(overlap).sum())}
    # Recompute HTTP wall-time aggregates directly from all 1,800 saved requests.
    for device in ['cuda','cpu']:
        web = pd.read_csv(SOURCE/f'03_result/result-web-http-raw-{device}.csv')
        summary = pd.read_csv(SOURCE/f'03_result/result-web-http-summary-{device}.csv')
        print('HTTP raw columns:', list(web.columns),flush=True)
        for row in summary.itertuples():
            group = web[web.method.eq(row.method)]
            check(f'{device} {row.method} 300 HTTP requests',len(group)==300)
            check(f'{device} {row.method} HTTP mean',np.isclose(group.http_ms.mean(),row.http_mean_ms))
            check(f'{device} {row.method} HTTP P95',np.isclose(np.percentile(group.http_ms,95),row.http_p95_ms))
    (output/'audit-checks.json').write_text(json.dumps({'diagnostics':diagnostics,'checks':checks},indent=2),encoding='utf-8')
    print(summaries.to_string(index=False))
    print(json.dumps(diagnostics,indent=2))
    print('Checks:',len(checks),'Failures:',sum(not x['pass'] for x in checks))
    for c in checks:
        if not c['pass']: print('FAIL',c)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
