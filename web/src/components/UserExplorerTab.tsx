'use client';

import { useMemo, useState } from 'react';
import { CheckCircle2, Eye, EyeOff, ExternalLink, SlidersHorizontal, Star } from 'lucide-react';
import {
  asinUrl,
  computeUserMetrics,
  DemoData,
  getLiveBlendRecs,
} from '@/lib/types';
import { ProductImage } from './ProductImage';

type ExplorerModel = 'SVD' | 'Blend' | 'ItemKNN' | 'Popularity' | 'Random' | 'Compare (3)';

export function UserExplorerTab({ data }: { data: DemoData }) {
  const [tagFilter, setTagFilter] = useState<string>('All');
  const [selectedUserId, setSelectedUserId] = useState<number>(data.users[0]?.id ?? 0);
  const [model, setModel] = useState<ExplorerModel>('SVD');
  const [alpha, setAlpha] = useState<number>(data.meta.selected_knn_weight);
  const [showTruth, setShowTruth] = useState<boolean>(true);

  const tags = useMemo(() => ['All', ...Array.from(new Set(data.users.map((u) => u.tag)))], [data.users]);
  const filteredUsers = useMemo(
    () => data.users.filter((u) => tagFilter === 'All' || u.tag === tagFilter),
    [data.users, tagFilter]
  );

  const user = useMemo(
    () => filteredUsers.find((u) => u.id === selectedUserId) || filteredUsers[0] || data.users[0],
    [filteredUsers, selectedUserId, data.users]
  );

  const heldSet = useMemo(() => new Set(user.held_out), [user]);

  const activeRecs = useMemo(() => {
    if (model === 'Compare (3)') return [];
    return model === 'Blend' ? getLiveBlendRecs(user, alpha) : user.recs[model] || [];
  }, [model, user, alpha]);

  const activeMetrics = useMemo(
    () => computeUserMetrics(activeRecs.map((r) => r[0]), heldSet),
    [activeRecs, heldSet]
  );

  const maxScore = Math.max(...activeRecs.map((r) => Math.max(r[1], 0)), 1e-9);

  return (
    <div className="space-y-5">
      {/* Controls Card */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
        <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
          <div>
            <h2 className="text-base font-bold text-slate-900">Real Held-Out Test User Explorer</h2>
            <p className="text-xs text-slate-500">
              Inspect actual Amazon fashion shoppers (anonymized numeric IDs), their past purchases, and whether each model predicts their future &ge;4★ items.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setShowTruth((s) => !s)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100 cursor-pointer"
          >
            {showTruth ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
            {showTruth ? 'Hide Ground Truth' : 'Reveal Ground Truth'}
          </button>
        </div>

        <div className="flex flex-wrap gap-1.5 mb-3">
          {tags.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => {
                setTagFilter(t);
                const first = data.users.find((u) => t === 'All' || u.tag === t);
                if (first) setSelectedUserId(first.id);
              }}
              className={`px-3 py-1 rounded-full text-xs font-semibold transition cursor-pointer ${
                tagFilter === t
                  ? 'bg-teal-700 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <select
          value={user.id}
          onChange={(e) => setSelectedUserId(Number(e.target.value))}
          className="w-full max-w-xl rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-800 focus:border-teal-600 focus:outline-none"
        >
          {filteredUsers.map((u) => {
            const maxHits = Math.max(...Object.values(u.metrics).map((m) => m.hits));
            return (
              <option key={u.id} value={u.id}>
                User #{u.id} — {u.tag} ({u.train_count} train reviews · {u.held_out.length} test target
                {u.held_out.length > 1 ? 's' : ''} · max {maxHits} hit{maxHits !== 1 ? 's' : ''})
              </option>
            );
          })}
        </select>
      </div>

      {/* Two Column Explorer */}
      <div className="grid grid-cols-1 lg:grid-cols-[370px_1fr] gap-5 items-start">
        {/* Left: User Profile + Targets + History */}
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-xs">
            <div className="flex items-center justify-between gap-2 mb-2">
              <h3 className="text-sm font-bold text-slate-900">Test User #{user.id}</h3>
              <span className="rounded-full bg-teal-50 px-2.5 py-0.5 text-[11px] font-bold text-teal-800 border border-teal-200">
                {user.tag}
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-3">
              {user.train_count} training reviews ({user.positive_train_count} positive &ge;4★) &middot;{' '}
              {user.held_out.length} held-out future item(s)
            </p>

            {showTruth ? (
              <div>
                <div className="text-[11px] font-bold uppercase tracking-wider text-emerald-700 mb-2">
                  Held-Out Test Targets (Future &ge;4★)
                </div>
                <div className="space-y-2">
                  {user.held_out.map((asin) => {
                    const info = data.catalog[asin];
                    return (
                      <div
                        key={asin}
                        className="flex items-center gap-3 rounded-lg border border-emerald-200 bg-emerald-50/70 p-2.5"
                      >
                        <ProductImage
                          src={info?.img}
                          alt={info?.t || asin}
                          category={info?.c || 'Fashion & Apparel'}
                          className="h-14 w-14"
                        />
                        <div className="min-w-0 flex-1">
                          <div className="text-xs font-bold text-slate-900 line-clamp-2">
                            {info?.t || asin}
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                            <a
                              href={asinUrl(asin)}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="font-mono text-teal-700 hover:underline inline-flex items-center gap-0.5"
                            >
                              {asin} <ExternalLink className="h-2.5 w-2.5" />
                            </a>
                            {info?.pr ? <span className="font-semibold">${info.pr.toFixed(2)}</span> : null}
                            <span className="rounded bg-white px-1.5 py-0.5 border border-emerald-200 text-[10px]">
                              {info?.c}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="rounded-lg bg-slate-100 p-3 text-xs text-slate-600">
                Ground truth is hidden. Inspect the recommendations, then click <strong>Reveal Ground Truth</strong> to check hits!
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-xs">
            <h3 className="text-sm font-bold text-slate-900 mb-1">Training History (&ge;4★)</h3>
            <p className="text-xs text-slate-500 mb-3">Past products used to model this user&apos;s taste</p>
            <div className="space-y-2 max-h-[540px] overflow-y-auto pr-1">
              {user.history.map((h) => {
                const info = data.catalog[h.asin];
                return (
                  <div
                    key={h.asin}
                    className="flex items-center gap-2.5 rounded-lg border border-slate-200 bg-slate-50/50 p-2"
                  >
                    <ProductImage
                      src={info?.img}
                      alt={info?.t || h.asin}
                      category={info?.c || 'Fashion & Apparel'}
                      className="h-12 w-12"
                    />
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-semibold text-slate-800 line-clamp-2">
                        {info?.t || h.asin}
                      </div>
                      <div className="mt-0.5 flex items-center gap-2 text-[11px] text-slate-500">
                        <span className="inline-flex items-center gap-0.5 text-amber-600 font-semibold">
                          <Star className="h-3 w-3 fill-amber-400 text-amber-500" /> {h.rating}.0
                        </span>
                        <span>{h.date}</span>
                        <span className="font-mono">{h.asin}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: Recommendations */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
          <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-900">Top-10 Model Recommendations</h3>
              <p className="text-xs text-slate-500">Ranked over all 22,554 unseen products</p>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {(['SVD', 'Blend', 'ItemKNN', 'Popularity', 'Random', 'Compare (3)'] as ExplorerModel[]).map(
                (m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setModel(m)}
                    className={`px-3 py-1 rounded-full text-xs font-semibold transition cursor-pointer ${
                      model === m
                        ? 'bg-teal-700 text-white'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    {m}
                  </button>
                )
              )}
            </div>
          </div>

          {model === 'Blend' && (
            <div className="mb-4 rounded-lg border border-teal-200 bg-teal-50/60 p-3.5">
              <div className="flex items-center justify-between text-xs font-bold text-slate-800 mb-1.5">
                <span className="inline-flex items-center gap-1.5">
                  <SlidersHorizontal className="h-3.5 w-3.5 text-teal-700" />
                  Live Blend Weight (&alpha; &times; ItemKNN + (1 &minus; &alpha;) &times; Popularity)
                </span>
                <span className="rounded-md bg-teal-700 px-2 py-0.5 font-mono text-white">
                  &alpha; = {alpha.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full accent-teal-700 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-500 mt-1">
                <span>0.00 (Pure Popularity)</span>
                <span>0.50 (Validated Default)</span>
                <span>1.00 (Pure ItemKNN)</span>
              </div>
            </div>
          )}

          {model === 'Compare (3)' ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
              {(['SVD', 'Blend', 'Popularity'] as const).map((m) => {
                const recs = m === 'Blend' ? getLiveBlendRecs(user, alpha) : user.recs[m];
                const met = computeUserMetrics(recs.map((r) => r[0]), heldSet);
                return (
                  <div key={m} className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
                    <div className="flex items-center justify-between mb-2.5">
                      <span className="text-sm font-bold text-slate-900">
                        {m}
                        {m === 'Blend' ? ` (α=${alpha.toFixed(2)})` : ''}
                      </span>
                      {showTruth && (
                        <span
                          className={`rounded-full px-2 py-0.5 text-[11px] font-bold ${
                            met.hits > 0
                              ? 'bg-emerald-100 text-emerald-800'
                              : 'bg-slate-200 text-slate-600'
                          }`}
                        >
                          {met.hits} hit{met.hits !== 1 ? 's' : ''} · {met.ndcg_at_10.toFixed(3)}
                        </span>
                      )}
                    </div>
                    <div className="space-y-2">
                      {recs.map(([asin], idx) => {
                        const info = data.catalog[asin];
                        const isHit = showTruth && heldSet.has(asin);
                        return (
                          <div
                            key={asin}
                            className={`flex items-center gap-2 rounded-lg border p-2 ${
                              isHit
                                ? 'border-emerald-300 bg-emerald-50'
                                : 'border-slate-200 bg-white'
                            }`}
                          >
                            <span
                              className={`h-5 w-5 shrink-0 rounded text-[10px] font-bold grid place-items-center ${
                                isHit ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600'
                              }`}
                            >
                              {idx + 1}
                            </span>
                            <ProductImage
                              src={info?.img}
                              alt={info?.t || asin}
                              category={info?.c || 'Fashion & Apparel'}
                              className="h-10 w-10"
                            />
                            <div className="min-w-0 flex-1">
                              <div className="text-[11.5px] font-semibold text-slate-800 line-clamp-2 leading-snug">
                                {info?.t || asin}
                              </div>
                              {isHit && (
                                <span className="inline-flex items-center gap-0.5 text-[10px] font-bold text-emerald-700 mt-0.5">
                                  <CheckCircle2 className="h-3 w-3" /> HIT
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <>
              {showTruth && (
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mb-4">
                  {[
                    { k: 'Hits @10', v: `${activeMetrics.hits} / ${user.held_out.length}` },
                    { k: 'NDCG@10', v: activeMetrics.ndcg_at_10.toFixed(4) },
                    { k: 'Recall@10', v: activeMetrics.recall_at_10.toFixed(4) },
                    { k: 'Precision@10', v: activeMetrics.precision_at_10.toFixed(4) },
                    { k: 'MRR@10', v: activeMetrics.mrr_at_10.toFixed(4) },
                  ].map((m) => (
                    <div
                      key={m.k}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-center"
                    >
                      <div className="text-base font-extrabold text-slate-900 tabular-nums">{m.v}</div>
                      <div className="text-[11px] text-slate-500">{m.k}</div>
                    </div>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {activeRecs.map(([asin, score], idx) => {
                  const info = data.catalog[asin];
                  const isHit = showTruth && heldSet.has(asin);
                  const pct = Math.max(5, Math.min(100, (Math.max(score, 0) / maxScore) * 100));
                  return (
                    <div
                      key={asin}
                      className={`group flex gap-3 rounded-xl border p-3 transition ${
                        isHit
                          ? 'border-emerald-300 bg-emerald-50/80 ring-1 ring-emerald-200'
                          : 'border-slate-200 bg-white hover:border-teal-300'
                      }`}
                    >
                      <div className="relative">
                        <ProductImage
                          src={info?.img}
                          alt={info?.t || asin}
                          category={info?.c || 'Fashion & Apparel'}
                          className="h-20 w-20"
                        />
                        <span
                          className={`absolute -top-2 -left-2 h-6 w-6 rounded-full text-xs font-extrabold grid place-items-center shadow-xs ${
                            isHit ? 'bg-emerald-600 text-white' : 'bg-slate-900 text-white'
                          }`}
                        >
                          {idx + 1}
                        </span>
                      </div>

                      <div className="min-w-0 flex-1 flex flex-col justify-between">
                        <div>
                          <div className="flex items-start justify-between gap-1.5">
                            <a
                              href={asinUrl(asin)}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs font-bold text-slate-900 hover:text-teal-700 line-clamp-2 leading-snug"
                            >
                              {info?.t || asin}
                            </a>
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-500">
                            <span className="rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-700">
                              {info?.c}
                            </span>
                            {info?.pr ? (
                              <span className="font-bold text-slate-800">${info.pr.toFixed(2)}</span>
                            ) : null}
                            <span>★ {info?.r.toFixed(1)} ({info?.n})</span>
                          </div>
                        </div>

                        <div className="mt-2 flex items-center justify-between gap-2">
                          {isHit ? (
                            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-bold text-white">
                              <CheckCircle2 className="h-3 w-3" /> MATCHED TEST TARGET
                            </span>
                          ) : (
                            <span className="font-mono text-[10.5px] text-slate-400">{asin}</span>
                          )}
                          <div className="flex items-center gap-1.5">
                            <div className="w-12 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                              <div className="h-full bg-teal-600 rounded-full" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-[11px] font-bold text-slate-700 tabular-nums">
                              {score.toFixed(3)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
