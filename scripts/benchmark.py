"""Reproducible warm-user, warm-item top-10 recommendation experiment.

Use global time cutoffs, training-only features, full candidate ranking, and
validation-only blend selection. This is an offline pilot, not a sales claim.
"""
import argparse
import gzip
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix, diags, save_npz
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED, K = 42, 10


def top_k(scores, k=K):
    """Rank by score descending, breaking ties by ascending item index."""
    assert np.isfinite(scores).sum() >= k
    partial = np.argpartition(-scores, k - 1)[:k]
    cutoff = scores[partial].min()
    above = np.flatnonzero(scores > cutoff)
    equal = np.flatnonzero(scores == cutoff)[:k - len(above)]
    candidates = np.concatenate([above, equal])
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def metrics(recommended, relevant):
    hits = np.isin(recommended, list(relevant)).astype(float)
    discounts = 1 / np.log2(np.arange(2, K + 2))
    first = np.flatnonzero(hits)
    return {
        'precision_at_10': float(hits.sum() / K),
        'recall_at_10': float(hits.sum() / len(relevant)),
        'ndcg_at_10': float((hits * discounts).sum() / discounts[:min(K, len(relevant))].sum()),
        'mrr_at_10': float(1 / (first[0] + 1)) if len(first) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-users', type=int, default=3000, help='0 evaluates all eligible test users')
    parser.add_argument('--validation-users', type=int, default=1000, help='0 evaluates all eligible validation users')
    args = parser.parse_args()
    assert args.test_users >= 0 and args.validation_users >= 0
    started = time.perf_counter()
    rows = []
    with gzip.open(ROOT / 'data/raw/reviews_Clothing_Shoes_and_Jewelry_5.json.gz', 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            rows.append((record['reviewerID'], record['asin'], record['overall'], record['unixReviewTime']))
    data = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    raw_count = len(data)
    data = data.sort_values(['timestamp', 'user_id', 'item_id']).drop_duplicates(['user_id', 'item_id'], keep='first')
    # Date boundaries keep reviews on the same day in the same partition.
    train_cut, test_cut = [int(data.timestamp.quantile(q)) for q in (0.8, 0.9)]
    train = data[data.timestamp < train_cut].copy()
    validation = data[(data.timestamp >= train_cut) & (data.timestamp < test_cut)].copy()
    test = data[data.timestamp >= test_cut].copy()
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    for name, frame in [('train', train), ('validation', validation), ('test', test)]:
        frame.to_csv(ROOT / f'data/processed/{name}.csv', index=False)
    users = sorted(train.user_id.unique())
    positive_train = train[train.rating >= 4]
    items = sorted(positive_train.item_id.unique())
    uid, iid = {v: i for i, v in enumerate(users)}, {v: i for i, v in enumerate(items)}
    x = csr_matrix((np.ones(len(positive_train), dtype=np.float32),
                    (positive_train.user_id.map(uid), positive_train.item_id.map(iid))),
                   shape=(len(users), len(items)))
    save_npz(ROOT / 'data/processed/train_positive_interactions.npz', x)
    (ROOT / 'data/processed/index.json').write_text(json.dumps({'users': users, 'items': items}))
    popularity = np.asarray(x.sum(axis=0)).ravel()
    popularity /= popularity.max()
    history = train.groupby('user_id').size().to_dict()
    def make_eval(frame, prior, limit):
        positives = frame[frame.rating >= 4]
        warm = positives[positives.item_id.isin(iid) & positives.user_id.isin(uid)]
        warm = warm.assign(item_index=warm.item_id.map(iid))
        truth = {uid[u]: set(group.item_index.astype(int)) for u, group in warm.groupby('user_id') if history.get(u, 0) >= 3}
        eligible = sorted(truth)
        selected = eligible
        if limit and len(eligible) > limit:
            selected = sorted(np.random.default_rng(SEED).choice(eligible, limit, replace=False).tolist())
        known_prior = prior[prior.item_id.isin(iid) & prior.user_id.isin(uid)].copy()
        known_prior['item_index'] = known_prior.item_id.map(iid)
        seen = {uid[u]: set(group.item_index.astype(int))
                for u, group in known_prior.groupby('user_id')}
        for u in selected:
            assert not truth[u].intersection(seen[u])
        return selected, truth, seen, {'eligible_users': len(eligible), 'evaluated_users': len(selected),
            'positive_events': len(positives), 'warm_user_and_item_positive_events': len(warm)}
    val_users, val_truth, val_seen, val_stats = make_eval(validation, train, args.validation_users)
    test_users, test_truth, test_seen, test_stats = make_eval(test, pd.concat([train, validation]), args.test_users)
    assert val_users and test_users
    print(f'Prepared data: {len(users)} users, {len(items)} candidate items', flush=True)
    fit_times = {'Random': 0.0, 'Popularity': 0.0}
    begin = time.perf_counter()
    normalized = x @ diags(1 / np.sqrt(np.maximum(np.asarray(x.sum(axis=0)).ravel(), 1)))
    similarities = (normalized.T @ normalized).tocsr()
    similarities.setdiag(0)
    similarities.eliminate_zeros()
    # Retain the 100 largest cosine similarities per source item.
    for row in range(similarities.shape[0]):
        lo, hi = similarities.indptr[row:row + 2]
        if hi - lo > 100:
            values = similarities.data[lo:hi]
            columns = similarities.indices[lo:hi]
            chosen = np.lexsort((columns, -values))[:100]
            mask = np.ones(hi - lo, dtype=bool)
            mask[chosen] = False
            values[mask] = 0
    similarities.eliminate_zeros()
    fit_times['ItemKNN'] = time.perf_counter() - begin
    print(f'Fitted ItemKNN in {fit_times["ItemKNN"]:.2f}s', flush=True)
    begin = time.perf_counter()
    svd = TruncatedSVD(n_components=64, n_iter=7, random_state=SEED)
    user_factors = svd.fit_transform(x)
    item_factors = svd.components_
    fit_times['SVD'] = time.perf_counter() - begin
    print(f'Fitted SVD in {fit_times["SVD"]:.2f}s', flush=True)

    def base_scores(u):
        knn = (x.getrow(u) @ similarities).toarray().ravel()
        knn /= max(float(knn.max()), 1e-12)
        # Very small popularity tie-break/fallback for items with equal KNN scores.
        knn += 1e-8 * popularity
        latent = user_factors[u] @ item_factors
        return knn, latent

    validation_rows = []
    for alpha in (0.25, 0.5, 0.75):
        values = []
        for u in val_users:
            knn, _ = base_scores(u)
            score = alpha * knn + (1 - alpha) * popularity
            score[list(val_seen[u])] = -np.inf
            values.append(metrics(top_k(score), val_truth[u])['ndcg_at_10'])
        validation_rows.append({'knn_weight': alpha, 'ndcg_at_10': float(np.mean(values))})
        print(f'Validated blend weight {alpha}', flush=True)
    alpha = max(validation_rows, key=lambda row: row['ndcg_at_10'])['knn_weight']
    fit_times['Blend'] = fit_times['ItemKNN']
    all_rows, examples = [], []
    recommended_items = {name: set() for name in fit_times}
    inference = {name: 0.0 for name in fit_times}
    for n, u in enumerate(test_users):
        begin = time.perf_counter()
        knn, latent = base_scores(u)
        base_elapsed = time.perf_counter() - begin
        # Shared score construction time is reported separately, not misattributed.
        inference['ItemKNN'] += base_elapsed
        score_map = {
            'Random': np.random.default_rng(SEED + u).random(len(items)),
            'Popularity': popularity.copy(), 'ItemKNN': knn,
            'SVD': latent, 'Blend': alpha * knn + (1 - alpha) * popularity,
        }
        for model, score in score_map.items():
            score[list(test_seen[u])] = -np.inf
            rec = top_k(score)
            assert len(rec) == K and len(set(rec)) == K
            assert not set(rec).intersection(test_seen[u])
            assert np.isfinite(score[rec]).all()
            values = metrics(rec, test_truth[u])
            assert all(0 <= v <= 1 for v in values.values())
            all_rows.append({'model': model, 'user_index': u, **values})
            recommended_items[model].update(rec.tolist())
            if n < 3:
                examples.append({'user_index': u, 'model': model,
                    'recommended_product_ids': [items[i] for i in rec],
                    'held_out_relevant_product_ids': [items[i] for i in sorted(test_truth[u])]})
        if (n + 1) % 1000 == 0:
            print(f'Evaluated {n + 1}/{len(test_users)} users', flush=True)
    per_user = pd.DataFrame(all_rows)
    per_user.to_csv(ROOT / 'results/per_user_metrics.csv', index=False)
    result = per_user.groupby('model').mean(numeric_only=True).drop(columns='user_index')
    result['catalog_coverage'] = [len(recommended_items[m]) / len(items) for m in result.index]
    result['fit_seconds'] = [fit_times[m] for m in result.index]
    result = result.sort_values('ndcg_at_10', ascending=False)
    result.to_csv(ROOT / 'results/metrics.csv')
    paired = per_user.pivot(index='user_index', columns='model', values='ndcg_at_10')
    delta = (paired.Blend - paired.Popularity).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = [float(rng.choice(delta, len(delta), replace=True).mean()) for _ in range(2000)]
    lower, upper = np.quantile(boot, [0.025, 0.975])
    metadata = {
        'run_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'raw_reviews': raw_count, 'unique_user_item_pairs': len(data),
        'raw_unique_users': int(data.user_id.nunique()), 'raw_unique_items': int(data.item_id.nunique()),
        'date_range_utc': [str(pd.to_datetime(data.timestamp.min(), unit='s', utc=True)), str(pd.to_datetime(data.timestamp.max(), unit='s', utc=True))],
        'cutoffs_utc': [str(pd.to_datetime(t, unit='s', utc=True)) for t in (train_cut, test_cut)],
        'split_rows': {'train': len(train), 'validation': len(validation), 'test': len(test)},
        'training_users': len(users), 'candidate_items': len(items), 'positive_training_events': x.nnz,
        'validation': val_stats, 'test': test_stats,
        'validation_blend_search': validation_rows, 'selected_knn_weight': alpha,
        'blend_minus_popularity_ndcg_mean': float(delta.mean()),
        'blend_minus_popularity_ndcg_paired_bootstrap_95_ci': [float(lower), float(upper)],
        'protocol': 'Global chronological approximately 80/10/10; unique user-item pairs; training rating >=4 as implicit positives; >=3 training reviews per evaluation user; warm items only; full training-positive catalog; remove all earlier reviewed items; macro-average metrics; model frozen at training cutoff, including during test.',
        'hyperparameters': {'item_knn_neighbors': 100, 'svd_components': 64, 'svd_iterations': 7, 'k': K},
        'verification': 'Asserted strict temporal boundaries, no repeated user-item pairs across splits, disjoint relevant/seen sets, 10 unique unseen finite-score recommendations per user/model, and metric bounds.',
        'elapsed_seconds': time.perf_counter() - started,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform()},
    }
    (ROOT / 'results/run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (ROOT / 'results/recommendation_examples.json').write_text(json.dumps(examples, indent=2) + '\n')
    print(result.to_string(), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
