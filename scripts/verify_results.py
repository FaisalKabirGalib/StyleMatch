"""Check saved research evidence and independently recalculate aggregate metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import metrics, top_k

ROOT = Path(__file__).resolve().parents[1]
assert top_k(np.full(12, 5.0)).tolist() == list(range(10))
assert top_k(np.array([-np.inf, 2., 2., 1.]), k=2).tolist() == [1, 2]
known = metrics(np.arange(10), {0, 2, 12})
assert np.isclose(known['recall_at_10'], 2 / 3)
assert np.isclose(known['precision_at_10'], .2)
assert np.isclose(known['mrr_at_10'], 1)
assert np.isclose(known['ndcg_at_10'], 1.5 / (1 + 1 / np.log2(3) + .5))
assert all(v == 0 for v in metrics(np.arange(10), {11}).values())

manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
with (ROOT / 'data/raw/reviews_Clothing_Shoes_and_Jewelry_5.json.gz').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['sha256']
splits = [pd.read_csv(ROOT / f'data/processed/{s}.csv') for s in ('train', 'validation', 'test')]
for earlier, later in zip(splits, splits[1:]):
    assert earlier.timestamp.max() < later.timestamp.min()
combined = pd.concat(splits)
assert not combined.duplicated(['user_id', 'item_id']).any()
run = json.loads((ROOT / 'results/run.json').read_text())
assert len(combined) == run['unique_user_item_pairs']
assert manifest['reviews'] == run['raw_reviews'] == 278677

per_user = pd.read_csv(ROOT / 'results/per_user_metrics.csv')
summary = pd.read_csv(ROOT / 'results/metrics.csv').set_index('model')
names = ['precision_at_10', 'recall_at_10', 'ndcg_at_10', 'mrr_at_10']
actual = per_user.groupby('model')[names].mean()
assert set(actual.index) == {'Random', 'Popularity', 'ItemKNN', 'SVD', 'Blend'}
assert np.allclose(actual.loc[summary.index], summary[names], rtol=1e-10, atol=1e-12)
assert (per_user[names].ge(0) & per_user[names].le(1)).all().all()
assert not per_user.duplicated(['model', 'user_index']).any()
assert per_user.groupby('model').size().eq(run['test']['evaluated_users']).all()
assert per_user.groupby('user_index').size().eq(5).all()
assert run['selected_knn_weight'] == max(run['validation_blend_search'], key=lambda row: row['ndcg_at_10'])['knn_weight']
for filename in ['progress_report.html', 'summary.html', 'comparison.png', 'comparison.svg']:
    assert (ROOT / 'results' / filename).stat().st_size > 0
print('VERIFIED: source checksum and count; chronological, disjoint splits; ranking helpers; all five models on identical users; aggregate metrics independently reproduced; report artifacts present.')
