'use client';

import { useState } from 'react';
import { BasketItem, CatalogItem, CATEGORY_COLORS, LiveBasketRec } from '@/lib/types';
import { ProductImage } from './ProductImage';

export function StyleMap2D({
  items,
  catalog,
  selectedIndices,
  recs,
  onToggle,
}: {
  items: BasketItem[];
  catalog: Record<string, CatalogItem>;
  selectedIndices: number[];
  recs: LiveBasketRec[];
  onToggle: (idx: number) => void;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const selSet = new Set(selectedIndices);
  const recMap = new Map(recs.map((r, rank) => [r.localIdx, rank + 1]));

  const W = 560;
  const H = 310;
  const PAD = 28;
  const sx = (x: number) => PAD + ((x + 1) / 2) * (W - PAD * 2);
  const sy = (y: number) => H - PAD - ((y + 1) / 2) * (H - PAD * 2);

  const hoveredItem = hoveredIdx !== null ? items[hoveredIdx] : null;
  const hoveredCat = hoveredItem ? catalog[hoveredItem.asin] : null;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-xs">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h3 className="text-sm font-bold text-slate-900">
            2D SVD Latent Style Space (PCA of 64-D Item Factors)
          </h3>
          <p className="text-xs text-slate-500">
            Each dot is a real fashion product. Click any dot to add/remove it from your basket.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs font-medium text-slate-600">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-full bg-slate-900 ring-2 ring-teal-400" /> In Basket
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-full bg-emerald-500 ring-2 ring-emerald-200" /> Top-10 Rec
          </span>
        </div>
      </div>

      <div className="relative rounded-lg border border-slate-100 bg-slate-50/70 overflow-hidden">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block"
          role="img"
          aria-label="2D projection of 64-dimensional SVD fashion embeddings"
        >
          {/* subtle quadrant axes */}
          <line x1={W / 2} y1={10} x2={W / 2} y2={H - 10} stroke="#e2e8f0" strokeDasharray="3 3" />
          <line x1={10} y1={H / 2} x2={W - 10} y2={H / 2} stroke="#e2e8f0" strokeDasharray="3 3" />

          {/* connecting lines from basket centroid to top-5 recs */}
          {selectedIndices.length > 0 && (() => {
            const cx =
              selectedIndices.reduce((acc, idx) => acc + sx(items[idx].xy[0]), 0) /
              selectedIndices.length;
            const cy =
              selectedIndices.reduce((acc, idx) => acc + sy(items[idx].xy[1]), 0) /
              selectedIndices.length;
            return recs.slice(0, 5).map((r) => (
              <line
                key={`edge-${r.localIdx}`}
                x1={cx}
                y1={cy}
                x2={sx(items[r.localIdx].xy[0])}
                y2={sy(items[r.localIdx].xy[1])}
                stroke="#0f766e"
                strokeOpacity={0.28}
                strokeWidth={1.5}
                strokeDasharray="4 3"
              />
            ));
          })()}

          {/* regular points first, then recs, then selected so highlights stay on top */}
          {items.map((it, idx) => {
            const isSel = selSet.has(idx);
            const recRank = recMap.get(idx);
            if (isSel || recRank) return null;
            const info = catalog[it.asin];
            const color = CATEGORY_COLORS[info?.c || 'Fashion & Apparel'] || '#64748b';
            return (
              <circle
                key={it.asin}
                cx={sx(it.xy[0])}
                cy={sy(it.xy[1])}
                r={hoveredIdx === idx ? 6.5 : 4.2}
                fill={color}
                fillOpacity={0.62}
                stroke="#ffffff"
                strokeWidth={1}
                className="cursor-pointer transition-all"
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                onClick={() => onToggle(idx)}
              />
            );
          })}

          {recs.map((r, i) => {
            const it = items[r.localIdx];
            const cx = sx(it.xy[0]);
            const cy = sy(it.xy[1]);
            return (
              <g
                key={`rec-${it.asin}`}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredIdx(r.localIdx)}
                onMouseLeave={() => setHoveredIdx(null)}
                onClick={() => onToggle(r.localIdx)}
              >
                <circle cx={cx} cy={cy} r={8.5} fill="#10b981" fillOpacity={0.22} />
                <circle cx={cx} cy={cy} r={5.8} fill="#059669" stroke="#ffffff" strokeWidth={1.6} />
                <text
                  x={cx}
                  y={cy - 9}
                  textAnchor="middle"
                  fontSize="9"
                  fontWeight="700"
                  fill="#065f46"
                >
                  #{i + 1}
                </text>
              </g>
            );
          })}

          {selectedIndices.map((idx) => {
            const it = items[idx];
            const cx = sx(it.xy[0]);
            const cy = sy(it.xy[1]);
            return (
              <g
                key={`sel-${it.asin}`}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                onClick={() => onToggle(idx)}
              >
                <circle cx={cx} cy={cy} r={10.5} fill="#0f766e" fillOpacity={0.22} />
                <circle cx={cx} cy={cy} r={6.5} fill="#0f172a" stroke="#2dd4bf" strokeWidth={2.2} />
              </g>
            );
          })}
        </svg>

        {hoveredItem && hoveredCat && (
          <div className="pointer-events-none absolute bottom-2.5 left-2.5 right-2.5 sm:right-auto sm:max-w-sm rounded-lg border border-slate-200 bg-white/95 backdrop-blur p-2.5 shadow-md flex items-center gap-2.5">
            <ProductImage
              src={hoveredCat.img}
              alt={hoveredCat.t}
              category={hoveredCat.c}
              className="h-12 w-12"
            />
            <div className="min-w-0">
              <div className="text-xs font-bold text-slate-900 line-clamp-1">{hoveredCat.t}</div>
              <div className="text-[11px] text-slate-500 flex items-center gap-2 mt-0.5">
                <span className="font-mono">{hoveredItem.asin}</span>
                <span>&middot;</span>
                <span>{hoveredCat.c}</span>
                {hoveredCat.pr ? <span>&middot; ${hoveredCat.pr.toFixed(2)}</span> : null}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
