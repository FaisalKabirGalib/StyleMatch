"""Export compact, reproducible real-model artifacts + official Amazon 2014 metadata
for the Next.js static GitHub Pages interactive demo.

Uses the exact same chronological split, training matrix, ItemKNN similarity graph,
and 64-component TruncatedSVD factors as scripts/benchmark.py (seed 42).
No raw reviewer IDs, reviewer names, or held-out review text are exported.
"""
import ast
from collections import Counter, defaultdict
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import PCA, TruncatedSVD
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / 'web' / 'public' / 'data' / 'demo_data.json'
SEED, K = 42, 10

CATEGORY_RULES = [
    ('Watches', r'\b(watch|watches|timepiece|strap|chronograph|casio|timex|seiko|invicta|fossil)\b'),
    ('Running & Sneakers', r'\b(sneaker|sneakers|running|runner|trainer|athletic|tennis shoe|new balance|asics|nike|adidas|reebok|skechers|converse|vans)\b'),
    ('Boots', r'\b(boot|boots|bootie|booties|ugg|timberland|combat|chelsea)\b'),
    ('Sandals & Flats', r'\b(sandal|sandals|flip flop|flip-flop|flat|flats|loafer|loafers|slipper|slippers|clog|clogs|crocs|heel|heels|pump|pumps|wedge|wedges)\b'),
    ('Jewelry & Necklaces', r'\b(necklace|pendant|bracelet|bangle|chain|locket|choker|jewelry|charm|anklet)\b'),
    ('Rings & Earrings', r'\b(ring|rings|earring|earrings|stud|studs|hoop|hoops|diamond|sterling)\b'),
    ('Bags & Wallets', r'\b(bag|bags|purse|handbag|tote|backpack|wallet|clutch|crossbody|messenger|luggage|suitcase|duffel|briefcase)\b'),
    ('Sunglasses & Eyewear', r'\b(sunglass|sunglasses|glasses|eyewear|shades|aviator|polarized|frame|frames)\b'),
    ('Dresses & Skirts', r'\b(dress|dresses|skirt|skirts|gown|maxi|sundress|tunic|romper)\b'),
    ('Jackets & Coats', r'\b(jacket|jackets|coat|coats|hoodie|hoodies|blazer|cardigan|sweater|sweaters|vest|parka|fleece|pullover)\b'),
    ('Pants & Jeans', r'\b(jean|jeans|pant|pants|legging|leggings|short|shorts|trouser|trousers|slack|slacks|denim|capri|chino|levi)\b'),
    ('Shirts & Tops', r'\b(shirt|shirts|t-shirt|tee|blouse|top|tops|polo|tank|camisole|henley|jersey)\b'),
    ('Socks & Underwear', r'\b(sock|socks|bra|bras|underwear|boxer|boxers|brief|briefs|pantie|panties|pajama|pajamas|robe|lingerie|tights|hosiery|shapewear)\b'),
    ('Accessories & Hats', r'\b(belt|belts|hat|hats|cap|caps|beanie|scarf|scarves|glove|gloves|tie|ties|umbrella|keychain|headband)\b'),
]
COMPILED_RULES = [(cat, re.compile(pat, re.IGNORECASE)) for cat, pat in CATEGORY_RULES]

STOPWORDS = {
    'the', 'and', 'for', 'not', 'but', 'are', 'was', 'you', 'all', 'can', 'had', 'her',
    'his', 'she', 'him', 'one', 'our', 'out', 'get', 'got', 'has', 'too', 'its', 'who',
    'way', 'day', 'did', 'use', 'now', 'new', 'old', 'see', 'two', 'any', 'how', 'man',
    'boy', 'may', 'say', 'try', 'ask', 'men', 'put', 'end', 'why', 'let', 'big', 'off',
    'own', 'run', 'set', 'top', 'bit', 'far', 'few', 'yet', 'per', 'via', 'etc', 'dont',
    'cant', 'wont', 'didnt', 'doesnt', 'isnt', 'wasnt', 'ive', 'youre', 'theyre', 'thats',
    'great', 'good', 'nice', 'love', 'loved', 'very', 'really', 'just', 'well', 'best',
    'cute', 'perfect', 'awesome', 'excellent', 'amazing', 'comfortable', 'comfy', 'super',
    'like', 'fits', 'fit', 'size', 'small', 'large', 'little', 'much', 'more', 'most',
    'this', 'that', 'these', 'those', 'with', 'from', 'they', 'them', 'their', 'have',
    'been', 'were', 'what', 'when', 'where', 'will', 'would', 'could', 'should', 'about',
    'than', 'then', 'only', 'also', 'into', 'over', 'some', 'such', 'your', 'mine', 'ours',
    'price', 'worth', 'money', 'quality', 'product', 'item', 'bought', 'buy', 'ordered',
    'order', 'wear', 'wearing', 'worn', 'look', 'looks', 'looking', 'color', 'colors',
    'made', 'make', 'five', 'four', 'stars', 'star', 'happy', 'glad', 'recommend',
    'recommended', 'favorite', 'disappointed', 'cheap', 'pretty', 'beautiful', 'wonderful',
    'fantastic', 'husband', 'wife', 'daughter', 'son', 'gift', 'christmas', 'birthday',
    'work', 'works', 'every', 'ever', 'never', 'always', 'still', 'again', 'first', 'second',
    'third', 'pair', 'pairs', 'another', 'other', 'different', 'exact', 'exactly', 'true',
    'runs', 'tight', 'loose', 'long', 'short', 'tall', 'wide', 'narrow', 'heavy', 'light',
    'soft', 'hard', 'warm', 'cool', 'cold', 'winter', 'summer', 'spring', 'fall', 'year',
    'years', 'month', 'months', 'week', 'weeks', 'days', 'time', 'times', 'amazon', 'does',
    'didn', 'don', 'isn', 'won', 'wasn', 'aren', 'haven', 'hasn', 'hadn', 'couldn', 'wouldn',
    'shouldn', 'there', 'here', 'after', 'before', 'because', 'while', 'which', 'through',
    'under', 'without', 'around', 'though', 'although', 'since', 'until', 'during', 'between',
}


def top_k(scores, k=K):
    partial = np.argpartition(-scores, k - 1)[:k]
    cutoff = scores[partial].min()
    above = np.flatnonzero(scores > cutoff)
    equal = np.flatnonzero(scores == cutoff)[:k - len(above)]
    candidates = np.concatenate([above, equal])
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def compute_metrics(recommended, relevant):
    hits = np.isin(recommended, list(relevant)).astype(float)
    discounts = 1 / np.log2(np.arange(2, K + 2))
    first = np.flatnonzero(hits)
    return {
        'precision_at_10': round(float(hits.sum() / K), 5),
        'recall_at_10': round(float(hits.sum() / len(relevant)), 5),
        'ndcg_at_10': round(float((hits * discounts).sum() / discounts[:min(K, len(relevant))].sum()), 5),
        'mrr_at_10': round(float(1 / (first[0] + 1)) if len(first) else 0.0, 5),
        'hits': int(hits.sum()),
    }


def load_amazon_metadata(meta_path: Path, target_asins: set[str]):
    """Stream-filter official Amazon 2014 metadata for only the target ASINs."""
    meta_map = {}
    if not meta_path.exists():
        print(f'Metadata file not found at {meta_path}; using training-review hints only.', flush=True)
        return meta_map

    print(f'Streaming {meta_path.name} for {len(target_asins)} ASINs...', flush=True)
    with gzip.open(meta_path, 'rt', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            # Fast pre-check before parsing Python literal
            m = re.search(r"'asin':\s*'([^']+)'", line) or re.search(r'"asin":\s*"([^"]+)"', line)
            if not m or m.group(1) not in target_asins:
                continue
            try:
                rec = ast.literal_eval(line)
            except Exception:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
            asin = rec.get('asin')
            if not asin or asin not in target_asins:
                continue
            im_url = rec.get('imUrl') or ''
            if im_url.startswith('http://'):
                im_url = 'https://' + im_url[len('http://'):]
            title = (rec.get('title') or '').strip()
            brand = (rec.get('brand') or '').strip()
            price = rec.get('price')
            price_val = round(float(price), 2) if isinstance(price, (int, float)) and price > 0 else None
            # Clean title for display
            if len(title) > 110:
                title = title[:107].rsplit(' ', 1)[0] + '…'
            meta_map[asin] = {
                'title': title,
                'brand': brand,
                'price': price_val,
                'img': im_url,
            }
            if len(meta_map) >= len(target_asins):
                break
    print(f'Matched official Amazon metadata for {len(meta_map)}/{len(target_asins)} products.', flush=True)
    return meta_map


def infer_item_hints(raw_path: Path, train_cut: int, target_asins: set[str]):
    """Infer category and training review stats strictly from training-window reviews."""
    cat_votes = defaultdict(Counter)
    word_counts = defaultdict(Counter)
    ratings = defaultdict(list)

    with gzip.open(raw_path, 'rt') as stream:
        for line in stream:
            rec = json.loads(line)
            asin = rec['asin']
            if asin not in target_asins or rec['unixReviewTime'] >= train_cut:
                continue
            ratings[asin].append(float(rec['overall']))
            summary = rec.get('summary') or ''
            text = (summary + ' ' + (rec.get('reviewText') or '')[:220]).strip()
            for cat, regex in COMPILED_RULES:
                if regex.search(text):
                    cat_votes[asin][cat] += 2 if regex.search(summary) else 1
            for tok in re.findall(r"[a-zA-Z]{3,16}", summary.lower()):
                if tok not in STOPWORDS:
                    word_counts[asin][tok] += 1

    hints = {}
    for asin in target_asins:
        cat = cat_votes[asin].most_common(1)[0][0] if cat_votes[asin] else 'Fashion & Apparel'
        top_words = [w for w, _ in word_counts[asin].most_common(3)]
        rlist = ratings[asin]
        avg_r = round(float(np.mean(rlist)), 2) if rlist else 0.0
        fallback_label = f"{cat}: " + ', '.join(top_words) if top_words else cat
        hints[asin] = {
            'category': cat,
            'fallback_label': fallback_label,
            'train_reviews': len(rlist),
            'avg_rating': avg_r,
        }
    return hints


def main():
    raw_path = ROOT / 'data/raw/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
    meta_path = ROOT / 'data/raw/meta_Clothing_Shoes_and_Jewelry.json.gz'
    run_meta = json.loads((ROOT / 'results/run.json').read_text())
    metrics_df = pd.read_csv(ROOT / 'results/metrics.csv')

    rows = []
    with gzip.open(raw_path, 'rt') as stream:
        for line in stream:
            rec = json.loads(line)
            rows.append((rec['reviewerID'], rec['asin'], rec['overall'], rec['unixReviewTime']))
    data = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    data = data.sort_values(['timestamp', 'user_id', 'item_id']).drop_duplicates(['user_id', 'item_id'], keep='first')
    train_cut, test_cut = [int(data.timestamp.quantile(q)) for q in (0.8, 0.9)]
    train = data[data.timestamp < train_cut].copy()
    validation = data[(data.timestamp >= train_cut) & (data.timestamp < test_cut)].copy()
    test = data[data.timestamp >= test_cut].copy()

    users = sorted(train.user_id.unique())
    positive_train = train[train.rating >= 4]
    items = sorted(positive_train.item_id.unique())
    uid, iid = {v: i for i, v in enumerate(users)}, {v: i for i, v in enumerate(items)}

    x = csr_matrix(
        (np.ones(len(positive_train), dtype=np.float32),
         (positive_train.user_id.map(uid), positive_train.item_id.map(iid))),
        shape=(len(users), len(items)),
    )
    pop_counts = np.asarray(x.sum(axis=0)).ravel()
    popularity = pop_counts / pop_counts.max()

    history_counts = train.groupby('user_id').size().to_dict()
    positives_test = test[test.rating >= 4]
    warm_test = positives_test[positives_test.item_id.isin(iid) & positives_test.user_id.isin(uid)].copy()
    warm_test['item_index'] = warm_test.item_id.map(iid)
    test_truth = {
        uid[u]: set(g.item_index.astype(int))
        for u, g in warm_test.groupby('user_id')
        if history_counts.get(u, 0) >= 3
    }
    eligible = sorted(test_truth)
    test_users = sorted(np.random.default_rng(SEED).choice(eligible, 3000, replace=False).tolist())
    prior = pd.concat([train, validation])
    known_prior = prior[prior.item_id.isin(iid) & prior.user_id.isin(uid)].copy()
    known_prior['item_index'] = known_prior.item_id.map(iid)
    test_seen = {uid[u]: set(g.item_index.astype(int)) for u, g in known_prior.groupby('user_id')}

    # Exact ItemKNN
    normalized = x @ diags(1 / np.sqrt(np.maximum(pop_counts, 1)))
    similarities = (normalized.T @ normalized).tocsr()
    similarities.setdiag(0)
    similarities.eliminate_zeros()
    for row in range(similarities.shape[0]):
        lo, hi = similarities.indptr[row:row + 2]
        if hi - lo > 100:
            vals = similarities.data[lo:hi]
            cols = similarities.indices[lo:hi]
            chosen = np.lexsort((cols, -vals))[:100]
            mask = np.ones(hi - lo, dtype=bool)
            mask[chosen] = False
            vals[mask] = 0
    similarities.eliminate_zeros()

    # Exact 64-component TruncatedSVD
    svd = TruncatedSVD(n_components=64, n_iter=7, random_state=SEED)
    user_factors = svd.fit_transform(x)
    item_factors = svd.components_
    alpha = float(run_meta['selected_knn_weight'])

    per_user = pd.read_csv(ROOT / 'results/per_user_metrics.csv')
    piv = per_user.pivot(index='user_index', columns='model', values='ndcg_at_10')

    buckets = {
        'svd_unique': piv[(piv['SVD'] > 0) & (piv['Popularity'] == 0) & (piv['ItemKNN'] == 0)].index.tolist(),
        'knn_blend_hit': piv[(piv['Blend'] > 0) & (piv['ItemKNN'] > 0) & (piv['Popularity'] == 0)].index.tolist(),
        'multi_hit': piv[(piv['SVD'] > 0) & (piv['Blend'] > 0)].index.tolist(),
        'pop_hit': piv[(piv['Popularity'] > 0) & (piv['ItemKNN'] == 0)].index.tolist(),
        'zero_hit': piv[(piv['SVD'] == 0) & (piv['Blend'] == 0) & (piv['ItemKNN'] == 0) & (piv['Popularity'] == 0)].index.tolist(),
    }
    rng = np.random.default_rng(SEED)
    selected_specs = []
    quotas = [
        ('svd_unique', 'SVD Hit (Personalized)', 10),
        ('knn_blend_hit', 'ItemKNN & Blend Hit', 10),
        ('multi_hit', 'Multi-Model Hit', 8),
        ('pop_hit', 'Popularity Hit', 6),
        ('zero_hit', 'Hard Case (0 Hits)', 6),
    ]
    seen_u = set()
    for key, label, n in quotas:
        pool = [int(u) for u in buckets[key] if int(u) not in seen_u]
        chosen = sorted(rng.choice(pool, n, replace=False).tolist()) if len(pool) > n else pool
        for u in chosen:
            seen_u.add(u)
            selected_specs.append((u, label))

    sel_user_ids = {users[u] for u, _ in selected_specs}
    sel_train = positive_train[positive_train.user_id.isin(sel_user_ids)].sort_values('timestamp', ascending=False)
    user_hist_map = defaultdict(list)
    for row in sel_train.itertuples(index=False):
        user_hist_map[uid[row.user_id]].append({
            'asin': row.item_id,
            'rating': int(row.rating),
            'date': pd.to_datetime(row.timestamp, unit='s', utc=True).strftime('%Y-%m-%d'),
        })

    demo_users = []
    referenced_asins = set()
    for u, tag in selected_specs:
        knn = (x.getrow(u) @ similarities).toarray().ravel()
        knn /= max(float(knn.max()), 1e-12)
        knn += 1e-8 * popularity
        latent = user_factors[u] @ item_factors
        blend = alpha * knn + (1 - alpha) * popularity

        score_map = {
            'SVD': latent.copy(),
            'Blend': blend.copy(),
            'ItemKNN': knn.copy(),
            'Popularity': popularity.copy(),
            'Random': np.random.default_rng(SEED + u).random(len(items)),
        }
        seen_idx = list(test_seen[u])
        recs_out, metrics_out = {}, {}
        for model, sc in score_map.items():
            sc[seen_idx] = -np.inf
            rec_idx = top_k(sc, K)
            recs_out[model] = [[items[i], round(float(sc[i]), 5)] for i in rec_idx]
            metrics_out[model] = compute_metrics(rec_idx, test_truth[u])
            referenced_asins.update(items[i] for i in rec_idx)

        knn_unseen, pop_unseen = knn.copy(), popularity.copy()
        knn_unseen[seen_idx] = -np.inf
        pop_unseen[seen_idx] = -np.inf
        pool_idx = sorted(set(top_k(knn_unseen, 25).tolist()) | set(top_k(pop_unseen, 20).tolist()))
        blend_pool = [[items[i], round(float(knn[i]), 5), round(float(popularity[i]), 5)] for i in pool_idx]
        referenced_asins.update(items[i] for i in pool_idx)

        hist = user_hist_map[u][:15]
        held_out = [items[i] for i in sorted(test_truth[u])]
        referenced_asins.update(h['asin'] for h in hist)
        referenced_asins.update(held_out)

        demo_users.append({
            'id': u,
            'tag': tag,
            'train_count': int(history_counts[users[u]]),
            'positive_train_count': len(user_hist_map[u]),
            'history': hist,
            'held_out': held_out,
            'recs': recs_out,
            'metrics': metrics_out,
            'blend_pool': blend_pool,
        })

    # Balanced 180 showcase items across categories for Live Basket Tester + 2D SVD Style Map
    top_pop_idx = np.argsort(-pop_counts)[:450].tolist()
    top_pop_asins = {items[i] for i in top_pop_idx}
    all_target_asins = referenced_asins | top_pop_asins

    print(f'Inferring training-only stats for {len(all_target_asins)} products...', flush=True)
    hints = infer_item_hints(raw_path, train_cut, all_target_asins)
    amazon_meta = load_amazon_metadata(meta_path, all_target_asins)

    # Prefer items that have official titles & images when building the 180-item showcase basket
    by_cat = defaultdict(list)
    for idx in top_pop_idx:
        asin = items[idx]
        has_img = bool(amazon_meta.get(asin, {}).get('img'))
        by_cat[hints[asin]['category']].append((0 if has_img else 1, idx))
    for c in by_cat:
        by_cat[c].sort(key=lambda pair: pair[0])

    basket_idx = []
    cat_names = sorted(by_cat.keys())
    pos = 0
    while len(basket_idx) < 180:
        added = False
        for c in cat_names:
            if pos < len(by_cat[c]):
                basket_idx.append(by_cat[c][pos][1])
                added = True
                if len(basket_idx) >= 180:
                    break
        if not added:
            break
        pos += 1

    # Compute 2D PCA coordinates from the 64-D SVD item factors for the 180 showcase items
    basket_factors = item_factors[:, basket_idx].T  # (180, 64)
    coords_2d = PCA(n_components=2, random_state=SEED).fit_transform(basket_factors)
    # Normalize to [-1, 1] for clean SVG rendering
    c_min = coords_2d.min(axis=0)
    c_max = coords_2d.max(axis=0)
    coords_norm = 2.0 * (coords_2d - c_min) / np.maximum(c_max - c_min, 1e-9) - 1.0

    basket_set = set(basket_idx)
    basket_idx_to_local = {orig_i: local_i for local_i, orig_i in enumerate(basket_idx)}

    basket_items = []
    for local_i, orig_i in enumerate(basket_idx):
        row_start, row_end = similarities.indptr[orig_i], similarities.indptr[orig_i + 1]
        cols = similarities.indices[row_start:row_end]
        vals = similarities.data[row_start:row_end]
        neighbors = [
            [basket_idx_to_local[int(c_idx)], round(float(v), 4)]
            for c_idx, v in zip(cols, vals)
            if int(c_idx) in basket_set
        ]
        neighbors.sort(key=lambda pair: -pair[1])
        basket_items.append({
            'asin': items[orig_i],
            'pop': round(float(popularity[orig_i]), 5),
            'neighbors': neighbors[:25],
            'svd': [round(float(v), 4) for v in item_factors[:, orig_i]],
            'xy': [round(float(coords_norm[local_i, 0]), 4), round(float(coords_norm[local_i, 1]), 4)],
        })

    used_asins = referenced_asins | {items[i] for i in basket_idx}
    catalog = {}
    with_images = 0
    for asin in sorted(used_asins):
        h = hints[asin]
        am = amazon_meta.get(asin, {})
        idx = iid[asin]
        title = am.get('title') or h['fallback_label']
        img = am.get('img') or ''
        if img:
            with_images += 1
        entry = {
            'c': h['category'],
            't': title,
            'n': h['train_reviews'],
            'r': h['avg_rating'],
            'p': round(float(popularity[idx]), 5),
        }
        if am.get('brand'):
            entry['b'] = am['brand']
        if am.get('price') is not None:
            entry['pr'] = am['price']
        if img:
            entry['img'] = img
        catalog[asin] = entry

    payload = {
        'meta': run_meta,
        'metrics': metrics_df.to_dict(orient='records'),
        'users': demo_users,
        'basket_items': basket_items,
        'catalog': catalog,
        'metadata_stats': {
            'total_catalog': len(catalog),
            'with_official_images': with_images,
        },
    }

    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(payload, separators=(',', ':')))
    size_kb = WEB_DATA.stat().st_size / 1024
    print(f'Wrote {WEB_DATA} ({size_kb:.1f} KB) — {with_images}/{len(catalog)} products with official Amazon images.')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
