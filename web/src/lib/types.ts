export interface MetricRow {
  model: string;
  precision_at_10: number;
  recall_at_10: number;
  ndcg_at_10: number;
  mrr_at_10: number;
  catalog_coverage: number;
  fit_seconds: number;
}

export interface CatalogItem {
  c: string; // category
  t: string; // title (official Amazon title or training-review fallback)
  n: number; // training review count
  r: number; // avg training rating
  p: number; // normalized popularity
  b?: string; // brand
  pr?: number; // price
  img?: string; // official Amazon imUrl (https)
}

export interface UserHistoryItem {
  asin: string;
  rating: number;
  date: string;
}

export interface UserMetricSet {
  precision_at_10: number;
  recall_at_10: number;
  ndcg_at_10: number;
  mrr_at_10: number;
  hits: number;
}

export interface DemoUser {
  id: number;
  tag: string;
  train_count: number;
  positive_train_count: number;
  history: UserHistoryItem[];
  held_out: string[];
  recs: Record<string, [string, number][]>;
  metrics: Record<string, UserMetricSet>;
  blend_pool: [string, number, number][];
}

export interface BasketItem {
  asin: string;
  pop: number;
  neighbors: [number, number][];
  svd: number[];
  xy: [number, number];
}

export interface DemoData {
  meta: {
    run_utc: string;
    seed: number;
    raw_reviews: number;
    unique_user_item_pairs: number;
    raw_unique_users: number;
    raw_unique_items: number;
    date_range_utc: [string, string];
    cutoffs_utc: [string, string];
    split_rows: { train: number; validation: number; test: number };
    training_users: number;
    candidate_items: number;
    positive_training_events: number;
    validation: { eligible_users: number; evaluated_users: number; positive_events: number; warm_user_and_item_positive_events: number };
    test: { eligible_users: number; evaluated_users: number; positive_events: number; warm_user_and_item_positive_events: number };
    validation_blend_search: { knn_weight: number; ndcg_at_10: number }[];
    selected_knn_weight: number;
    blend_minus_popularity_ndcg_mean: number;
    blend_minus_popularity_ndcg_paired_bootstrap_95_ci: [number, number];
    protocol: string;
    hyperparameters: { item_knn_neighbors: number; svd_components: number; svd_iterations: number; k: number };
    verification: string;
    elapsed_seconds: number;
    environment: Record<string, string>;
  };
  metrics: MetricRow[];
  users: DemoUser[];
  basket_items: BasketItem[];
  catalog: Record<string, CatalogItem>;
  metadata_stats?: {
    total_catalog: number;
    with_official_images: number;
  };
}

export const MODEL_COLORS: Record<string, string> = {
  SVD: '#0f766e',
  Blend: '#0284c7',
  ItemKNN: '#7c3aed',
  Popularity: '#d97706',
  Random: '#64748b',
};

export const CATEGORY_COLORS: Record<string, string> = {
  Watches: '#0284c7',
  'Running & Sneakers': '#0f766e',
  Boots: '#92400e',
  'Sandals & Flats': '#ea580c',
  'Jewelry & Necklaces': '#9333ea',
  'Rings & Earrings': '#db2777',
  'Bags & Wallets': '#4f46e5',
  'Sunglasses & Eyewear': '#0891b2',
  'Dresses & Skirts': '#e11d48',
  'Jackets & Coats': '#334155',
  'Pants & Jeans': '#2563eb',
  'Shirts & Tops': '#16a34a',
  'Socks & Underwear': '#65a30d',
  'Accessories & Hats': '#ca8a04',
  'Fashion & Apparel': '#64748b',
};

export function asinUrl(asin: string): string {
  return `https://www.amazon.com/dp/${encodeURIComponent(asin)}`;
}

export function computeUserMetrics(recAsins: string[], heldOut: Set<string>): UserMetricSet {
  const k = 10;
  let hits = 0;
  let dcg = 0;
  let mrr = 0;
  recAsins.forEach((a, i) => {
    if (heldOut.has(a)) {
      hits += 1;
      dcg += 1 / Math.log2(i + 2);
      if (!mrr) mrr = 1 / (i + 1);
    }
  });
  const idcgCount = Math.min(k, heldOut.size);
  let idcg = 0;
  for (let i = 0; i < idcgCount; i++) idcg += 1 / Math.log2(i + 2);
  return {
    hits,
    precision_at_10: hits / k,
    recall_at_10: heldOut.size ? hits / heldOut.size : 0,
    ndcg_at_10: idcg ? dcg / idcg : 0,
    mrr_at_10: mrr,
  };
}

export function getLiveBlendRecs(user: DemoUser, alpha: number): [string, number][] {
  const scored = user.blend_pool.map(([asin, knn, pop]) => ({
    asin,
    score: alpha * knn + (1 - alpha) * pop,
  }));
  scored.sort((a, b) => b.score - a.score || a.asin.localeCompare(b.asin));
  return scored.slice(0, 10).map((s) => [s.asin, s.score]);
}

export interface LiveBasketRec {
  localIdx: number;
  asin: string;
  score: number;
  reason: string;
}

export function scoreBasket(
  items: BasketItem[],
  selectedIndices: number[],
  model: 'SVD' | 'Blend' | 'ItemKNN' | 'Popularity',
  alpha: number
): LiveBasketRec[] {
  if (!selectedIndices.length) return [];
  const selSet = new Set(selectedIndices);
  const n = items.length;

  const knnRaw = new Float64Array(n);
  const bestDriver: ({ asin: string; sim: number } | null)[] = new Array(n).fill(null);
  for (const bIdx of selectedIndices) {
    const bAsin = items[bIdx].asin;
    for (const [nbIdx, sim] of items[bIdx].neighbors) {
      if (selSet.has(nbIdx)) continue;
      knnRaw[nbIdx] += sim;
      if (!bestDriver[nbIdx] || sim > bestDriver[nbIdx]!.sim) {
        bestDriver[nbIdx] = { asin: bAsin, sim };
      }
    }
  }
  let maxKnn = 1e-12;
  for (let i = 0; i < n; i++) {
    if (!selSet.has(i) && knnRaw[i] > maxKnn) maxKnn = knnRaw[i];
  }
  const knnNorm = new Float64Array(n);
  for (let i = 0; i < n; i++) knnNorm[i] = knnRaw[i] / maxKnn + 1e-8 * items[i].pop;

  const dim = items[0].svd.length;
  const uVec = new Float64Array(dim);
  for (const bIdx of selectedIndices) {
    const vec = items[bIdx].svd;
    for (let d = 0; d < dim; d++) uVec[d] += vec[d];
  }
  const svdScores = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (selSet.has(i)) continue;
    const vec = items[i].svd;
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += uVec[d] * vec[d];
    svdScores[i] = dot;
  }

  const scored: LiveBasketRec[] = [];
  for (let i = 0; i < n; i++) {
    if (selSet.has(i)) continue;
    let score = 0;
    let reason = '';
    if (model === 'SVD') {
      score = svdScores[i];
      reason = '64-D SVD latent alignment';
    } else if (model === 'ItemKNN') {
      score = knnNorm[i];
      reason = bestDriver[i]
        ? `Co-reviewed with ${bestDriver[i]!.asin} (cos ${bestDriver[i]!.sim.toFixed(2)})`
        : 'Popularity tie-break';
    } else if (model === 'Blend') {
      score = alpha * knnNorm[i] + (1 - alpha) * items[i].pop;
      reason = bestDriver[i]
        ? `KNN ${bestDriver[i]!.asin} (cos ${bestDriver[i]!.sim.toFixed(2)}) + Pop`
        : `Popularity weight (${Math.round((1 - alpha) * 100)}%)`;
    } else {
      score = items[i].pop;
      reason = 'Global training popularity';
    }
    scored.push({ localIdx: i, asin: items[i].asin, score, reason });
  }
  scored.sort((a, b) => b.score - a.score || a.asin.localeCompare(b.asin));
  return scored.slice(0, 10);
}
