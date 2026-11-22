'use client';

import { useState } from 'react';
import { Award, Database, Layers, Users } from 'lucide-react';
import { DemoData, MetricRow, MODEL_COLORS } from '@/lib/types';

type MetricKey = keyof Omit<MetricRow, 'model'>;

const METRIC_META: Record<MetricKey, { label: string; fmt: (v: number) => string; desc: string }> = {
  ndcg_at_10: {
    label: 'NDCG@10',
    fmt: (v) => v.toFixed(5),
    desc: 'Normalized Discounted Cumulative Gain at 10 — rewards placing relevant held-out products near the top.',
  },
  recall_at_10: {
    label: 'Recall@10',
    fmt: (v) => v.toFixed(5),
    desc: 'Fraction of a user’s held-out relevant products recovered in the top 10 recommendations.',
  },
  precision_at_10: {
    label: 'Precision@10',
    fmt: (v) => v.toFixed(5),
    desc: 'Fraction of the 10 recommended products actually rated ≥4 in the future test window.',
  },
  mrr_at_10: {
    label: 'MRR@10',
    fmt: (v) => v.toFixed(5),
    desc: 'Mean Reciprocal Rank — measures how high the first relevant hit appears.',
  },
  catalog_coverage: {
    label: 'Coverage',
    fmt: (v) => `${(v * 100).toFixed(2)}%`,
    desc: 'Share of the 22,554 candidate products recommended to at least one test user.',
  },
  fit_seconds: {
    label: 'Fit Time',
    fmt: (v) => `${v.toFixed(2)}s`,
    desc: 'Wall-clock training time on the frozen 222,413-review training split.',
  },
};

export function OverviewTab({ data }: { data: DemoData }) {
  const [metric, setMetric] = useState<MetricKey>('ndcg_at_10');
  const { meta, metrics } = data;
  const best = metrics[0];
  const pop = metrics.find((m) => m.model === 'Popularity')!;
  const lift = (best.ndcg_at_10 / pop.ndcg_at_10 - 1) * 100;

  const sorted = [...metrics].sort((a, b) => b[metric] - a[metric]);
  const maxVal = Math.max(...sorted.map((r) => r[metric]), 1e-9);

  // Scatter dimensions
  const W = 480;
  const H = 230;
  const padL = 58;
  const padR = 24;
  const padT = 18;
  const padB = 38;
  const maxX = 0.82;
  const maxY = 0.0068;
  const sx = (x: number) => padL + (x / maxX) * (W - padL - padR);
  const sy = (y: number) => H - padB - (y / maxY) * (H - padT - padB);

  const [ciLo, ciHi] = meta.blend_minus_popularity_ndcg_paired_bootstrap_95_ci;
  const maxValNdcg = Math.max(...meta.validation_blend_search.map((v) => v.ndcg_at_10), 1e-9);

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          {
            icon: Database,
            value: meta.raw_reviews.toLocaleString(),
            label: 'Source Reviews',
            sub: `${meta.raw_unique_users.toLocaleString()} users · ${meta.raw_unique_items.toLocaleString()} items`,
          },
          {
            icon: Layers,
            value: meta.candidate_items.toLocaleString(),
            label: 'Candidate Products',
            sub: `${meta.positive_training_events.toLocaleString()} positive training events`,
          },
          {
            icon: Users,
            value: meta.test.evaluated_users.toLocaleString(),
            label: 'Evaluated Test Users',
            sub: `Sampled from ${meta.test.eligible_users.toLocaleString()} eligible (seed ${meta.seed})`,
          },
          {
            icon: Award,
            value: `${best.model} (${best.ndcg_at_10.toFixed(5)})`,
            label: 'Top NDCG@10 Model',
            sub: `+${lift.toFixed(1)}% vs Popularity baseline`,
          },
        ].map((k) => {
          const Icon = k.icon;
          return (
            <div key={k.label} className="rounded-xl border border-slate-200 bg-white p-4 shadow-xs">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {k.label}
                </span>
                <Icon className="h-4 w-4 text-teal-600" />
              </div>
              <div className="text-2xl font-extrabold text-slate-900 tabular-nums tracking-tight">
                {k.value}
              </div>
              <div className="text-xs font-medium text-teal-700 mt-1">{k.sub}</div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
          <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-900">Model Comparison (Top-10)</h2>
              <p className="text-xs text-slate-500">Full-catalog ranking across 3,000 held-out test users</p>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {(Object.keys(METRIC_META) as MetricKey[]).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setMetric(k)}
                  className={`px-2.5 py-1 rounded-full text-xs font-semibold transition cursor-pointer ${
                    metric === k
                      ? 'bg-teal-700 text-white'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {METRIC_META[k].label}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            {sorted.map((r, idx) => {
              const pct = Math.max((r[metric] / maxVal) * 100, r[metric] > 0 ? 1.5 : 0);
              const color = idx === 0 ? '#0f766e' : MODEL_COLORS[r.model] || '#64748b';
              return (
                <div key={r.model} className="grid grid-cols-[92px_1fr_76px] items-center gap-3">
                  <div className="text-sm font-semibold text-slate-800">{r.model}</div>
                  <div className="h-6 rounded-md bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-md transition-all duration-300"
                      style={{ width: `${pct}%`, backgroundColor: color, opacity: idx === 0 ? 1 : 0.72 }}
                    />
                  </div>
                  <div className="text-right text-sm font-bold text-slate-800 tabular-nums">
                    {METRIC_META[metric].fmt(r[metric])}
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-slate-500 mt-4">{METRIC_META[metric].desc}</p>
        </div>

        {/* Accuracy vs Coverage Scatter */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
          <h2 className="text-base font-bold text-slate-900">Accuracy vs. Catalog Coverage</h2>
          <p className="text-xs text-slate-500 mb-3">
            NDCG@10 vs. fraction of 22,554 candidate products recommended
          </p>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
            {[0, 1, 2, 3, 4].map((i) => {
              const xv = (maxX / 4) * i;
              return (
                <g key={`gx-${i}`}>
                  <line x1={sx(xv)} y1={padT} x2={sx(xv)} y2={H - padB} stroke="#e2e8f0" strokeDasharray="3 3" />
                  <text x={sx(xv)} y={H - 14} textAnchor="middle" fill="#64748b" fontSize="10.5">
                    {Math.round(xv * 100)}%
                  </text>
                </g>
              );
            })}
            {[0, 1, 2, 3].map((i) => {
              const yv = (maxY / 3) * i;
              return (
                <g key={`gy-${i}`}>
                  <line x1={padL} y1={sy(yv)} x2={W - padR} y2={sy(yv)} stroke="#e2e8f0" strokeDasharray="3 3" />
                  <text x={padL - 8} y={sy(yv) + 3} textAnchor="end" fill="#64748b" fontSize="10.5">
                    {yv.toFixed(4)}
                  </text>
                </g>
              );
            })}
            <text x={W / 2} y={H - 1} textAnchor="middle" fill="#64748b" fontSize="11">
              Catalog Coverage (% of catalog recommended)
            </text>
            <text x={13} y={H / 2} textAnchor="middle" fill="#64748b" fontSize="11" transform={`rotate(-90 13 ${H / 2})`}>
              NDCG@10
            </text>
            {metrics.map((r) => {
              const cx = sx(r.catalog_coverage);
              const cy = sy(r.ndcg_at_10);
              const col = MODEL_COLORS[r.model] || '#0f766e';
              const dx = r.catalog_coverage > 0.6 ? -10 : 10;
              const anchor = r.catalog_coverage > 0.6 ? 'end' : 'start';
              const dy = r.model === 'Popularity' ? -10 : 4;
              return (
                <g key={r.model}>
                  <circle cx={cx} cy={cy} r={7} fill={col} stroke="#ffffff" strokeWidth={2} />
                  <text x={cx + dx} y={cy + dy} textAnchor={anchor} fill="#0f172a" fontSize="11.5" fontWeight="700">
                    {r.model}
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="text-xs text-slate-600 mt-3">
            <strong>Tradeoff:</strong> <strong>SVD</strong> wins NDCG@10 (0.00586) at 2.1% coverage, while{' '}
            <strong>Blend</strong> and <strong>ItemKNN</strong> surface 42%–58% of the catalog and still beat Popularity.
          </p>
        </div>
      </div>

      {/* Table + Validation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs overflow-x-auto">
          <h2 className="text-base font-bold text-slate-900">Full Measured Test Table</h2>
          <p className="text-xs text-slate-500 mb-3">Exact metrics from results/metrics.csv (fractions)</p>
          <table className="w-full text-sm tabular-nums border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase text-slate-500">
                <th className="py-2 text-left">Model</th>
                <th className="py-2 text-right">P@10</th>
                <th className="py-2 text-right">R@10</th>
                <th className="py-2 text-right">NDCG@10</th>
                <th className="py-2 text-right">MRR@10</th>
                <th className="py-2 text-right">Cov.</th>
                <th className="py-2 text-right">Fit</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((r, idx) => (
                <tr
                  key={r.model}
                  className={`border-b border-slate-100 ${idx === 0 ? 'bg-teal-50/70 font-semibold' : ''}`}
                >
                  <td className="py-2.5 text-left text-slate-900">{r.model}</td>
                  <td className="py-2.5 text-right">{r.precision_at_10.toFixed(5)}</td>
                  <td className="py-2.5 text-right">{r.recall_at_10.toFixed(5)}</td>
                  <td className="py-2.5 text-right">{r.ndcg_at_10.toFixed(5)}</td>
                  <td className="py-2.5 text-right">{r.mrr_at_10.toFixed(5)}</td>
                  <td className="py-2.5 text-right">{(r.catalog_coverage * 100).toFixed(2)}%</td>
                  <td className="py-2.5 text-right">{r.fit_seconds.toFixed(2)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
          <h2 className="text-base font-bold text-slate-900">Validation Blend Search &amp; Bootstrap CI</h2>
          <p className="text-xs text-slate-500 mb-4">
            Blend weight &alpha; selected strictly on {meta.validation.evaluated_users.toLocaleString()} validation users
          </p>
          <div className="space-y-3">
            {meta.validation_blend_search.map((s) => {
              const isBest = Math.abs(s.knn_weight - meta.selected_knn_weight) < 1e-6;
              return (
                <div key={s.knn_weight} className="grid grid-cols-[90px_1fr_76px] items-center gap-3">
                  <div className="text-sm font-semibold text-slate-800">
                    &alpha; = {s.knn_weight.toFixed(2)} {isBest ? '★' : ''}
                  </div>
                  <div className="h-5 rounded-md bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-md"
                      style={{
                        width: `${(s.ndcg_at_10 / maxValNdcg) * 100}%`,
                        backgroundColor: isBest ? '#0f766e' : '#94a3b8',
                      }}
                    />
                  </div>
                  <div className="text-right text-sm font-bold tabular-nums">{s.ndcg_at_10.toFixed(5)}</div>
                </div>
              );
            })}
          </div>
          <div className="mt-5 rounded-lg border-l-4 border-teal-600 bg-teal-50/70 p-3.5 text-xs text-slate-700 leading-relaxed">
            <strong className="text-slate-900">Paired Bootstrap (Blend &minus; Popularity NDCG@10):</strong>{' '}
            Mean difference{' '}
            <code className="font-mono font-bold text-teal-800">
              +{meta.blend_minus_popularity_ndcg_mean.toFixed(5)}
            </code>{' '}
            &middot; 95% CI (2,000 resamples):{' '}
            <code className="font-mono font-bold text-teal-800">
              [{ciLo.toFixed(5)}, {ciHi.toFixed(5)}]
            </code>
            . Because the interval slightly spans zero, this fixed pilot shows a positive trend without claiming universal statistical dominance.
          </div>
        </div>
      </div>
    </div>
  );
}
