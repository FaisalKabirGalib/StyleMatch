'use client';

import { useMemo, useState } from 'react';
import {
  Check,
  ExternalLink,
  Plus,
  Search,
  ShoppingBag,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import { asinUrl, DemoData, scoreBasket } from '@/lib/types';
import { ProductImage } from './ProductImage';
import { StyleMap2D } from './StyleMap2D';

type LiveModel = 'SVD' | 'Blend' | 'ItemKNN' | 'Popularity';

export function BasketStudioTab({ data }: { data: DemoData }) {
  const findPreset = (cats: string[]) => {
    const out: number[] = [];
    for (const cat of cats) {
      const idx = data.basket_items.findIndex(
        (it, i) => data.catalog[it.asin]?.c === cat && !out.includes(i)
      );
      if (idx >= 0) out.push(idx);
    }
    return out;
  };

  const [selected, setSelected] = useState<number[]>(() =>
    findPreset(['Running & Sneakers', 'Socks & Underwear', 'Pants & Jeans'])
  );
  const [model, setModel] = useState<LiveModel>('SVD');
  const [alpha, setAlpha] = useState<number>(data.meta.selected_knn_weight);
  const [search, setSearch] = useState('');
  const [catFilter, setCatFilter] = useState('All');

  const categories = useMemo(
    () => [
      'All',
      ...Array.from(new Set(data.basket_items.map((it) => data.catalog[it.asin]?.c || 'Other'))).sort(),
    ],
    [data]
  );

  const filteredItems = useMemo(() => {
    const q = search.trim().toLowerCase();
    return data.basket_items
      .map((it, idx) => ({ it, idx, info: data.catalog[it.asin] }))
      .filter(({ it, info }) => {
        if (catFilter !== 'All' && info?.c !== catFilter) return false;
        if (!q) return true;
        return (
          it.asin.toLowerCase().includes(q) ||
          (info?.t || '').toLowerCase().includes(q) ||
          (info?.c || '').toLowerCase().includes(q) ||
          (info?.b || '').toLowerCase().includes(q)
        );
      });
  }, [data, search, catFilter]);

  const recs = useMemo(
    () => scoreBasket(data.basket_items, selected, model, alpha),
    [data.basket_items, selected, model, alpha]
  );

  const maxScore = Math.max(...recs.map((r) => Math.max(r.score, 0)), 1e-9);
  const selSet = new Set(selected);

  const toggleItem = (idx: number) => {
    setSelected((prev) => {
      if (prev.includes(idx)) return prev.filter((x) => x !== idx);
      if (prev.length >= 6) return prev;
      return [...prev, idx];
    });
  };

  return (
    <div className="space-y-6">
      {/* Top Banner + Active Basket */}
      <div className="rounded-xl border border-teal-200 bg-gradient-to-r from-teal-50 via-white to-sky-50 p-5 shadow-xs">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-1.5 rounded-full bg-teal-700 px-2.5 py-0.5 text-[11px] font-bold text-white mb-2">
              <Sparkles className="h-3 w-3" /> LIVE IN-BROWSER INFERENCE
            </div>
            <h2 className="text-lg font-extrabold text-slate-900">
              Interactive Fashion Recommender Studio
            </h2>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Pick 1 to 6 real Amazon fashion products (or try a preset) to act as a shopper. Recommendations update instantly using the actual trained <strong>64-D TruncatedSVD</strong> factors and <strong>ItemKNN cosine similarity graph</strong>.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() =>
                setSelected(findPreset(['Running & Sneakers', 'Socks & Underwear', 'Pants & Jeans']))
              }
              className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:border-teal-500 cursor-pointer"
            >
              Preset: Athletic Runner
            </button>
            <button
              type="button"
              onClick={() =>
                setSelected(findPreset(['Watches', 'Bags & Wallets', 'Sunglasses & Eyewear']))
              }
              className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:border-teal-500 cursor-pointer"
            >
              Preset: Watches &amp; Travel
            </button>
            <button
              type="button"
              onClick={() =>
                setSelected(findPreset(['Dresses & Skirts', 'Sandals & Flats', 'Jewelry & Necklaces']))
              }
              className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:border-teal-500 cursor-pointer"
            >
              Preset: Evening &amp; Dress
            </button>
          </div>
        </div>

        {/* Selected Basket Strip */}
        <div className="mt-4 pt-4 border-t border-teal-100">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold text-slate-700 inline-flex items-center gap-1.5">
              <ShoppingBag className="h-3.5 w-3.5 text-teal-700" />
              Your Style Basket ({selected.length} / 6)
            </span>
            {selected.length > 0 && (
              <button
                type="button"
                onClick={() => setSelected([])}
                className="text-xs font-semibold text-slate-500 hover:text-rose-600 inline-flex items-center gap-1 cursor-pointer"
              >
                <Trash2 className="h-3 w-3" /> Clear all
              </button>
            )}
          </div>

          {selected.length === 0 ? (
            <div className="text-xs text-slate-500 italic">
              Click any product card below or click a dot on the 2D Style Map to add items.
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {selected.map((idx) => {
                const it = data.basket_items[idx];
                const info = data.catalog[it.asin];
                return (
                  <div
                    key={it.asin}
                    className="flex items-center gap-2 rounded-lg border border-teal-300 bg-white pl-1.5 pr-2.5 py-1 shadow-2xs"
                  >
                    <ProductImage
                      src={info?.img}
                      alt={info?.t || it.asin}
                      category={info?.c || 'Fashion & Apparel'}
                      className="h-8 w-8"
                    />
                    <span className="text-xs font-semibold text-slate-800 max-w-[180px] truncate">
                      {info?.t || it.asin}
                    </span>
                    <button
                      type="button"
                      onClick={() => toggleItem(idx)}
                      className="text-slate-400 hover:text-rose-600 cursor-pointer"
                      aria-label="Remove item"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-[440px_1fr] gap-6 items-start">
        {/* Left: Visual Product Picker */}
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-xs">
          <h3 className="text-sm font-bold text-slate-900 mb-3">
            1. Browse 180 Real Fashion Products
          </h3>
          <div className="flex gap-2 mb-3">
            <div className="relative flex-1">
              <Search className="h-3.5 w-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search brand, item, ASIN..."
                className="w-full rounded-lg border border-slate-300 pl-8 pr-3 py-1.5 text-xs focus:border-teal-600 focus:outline-none"
              />
            </div>
            <select
              value={catFilter}
              onChange={(e) => setCatFilter(e.target.value)}
              className="rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:border-teal-600 focus:outline-none max-w-[145px]"
            >
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-2 max-h-[680px] overflow-y-auto pr-1">
            {filteredItems.map(({ it, idx, info }) => {
              const isSel = selSet.has(idx);
              return (
                <button
                  key={it.asin}
                  type="button"
                  onClick={() => toggleItem(idx)}
                  className={`w-full text-left flex items-center gap-3 rounded-lg border p-2 transition cursor-pointer ${
                    isSel
                      ? 'border-teal-600 bg-teal-50/70 ring-1 ring-teal-500/30'
                      : 'border-slate-200 bg-white hover:border-teal-300'
                  }`}
                >
                  <ProductImage
                    src={info?.img}
                    alt={info?.t || it.asin}
                    category={info?.c || 'Fashion & Apparel'}
                    className="h-14 w-14"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-bold text-slate-900 line-clamp-2 leading-snug">
                      {info?.t || it.asin}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-500">
                      <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-700">
                        {info?.c}
                      </span>
                      {info?.pr ? (
                        <span className="font-bold text-slate-800">${info.pr.toFixed(2)}</span>
                      ) : null}
                      <span>★ {info?.r.toFixed(1)}</span>
                    </div>
                  </div>
                  <div
                    className={`h-7 w-7 shrink-0 rounded-full grid place-items-center ${
                      isSel ? 'bg-teal-700 text-white' : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {isSel ? <Check className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right: 2D Style Map + Visual Recommendation Grid */}
        <div className="space-y-5">
          <StyleMap2D
            items={data.basket_items}
            catalog={data.catalog}
            selectedIndices={selected}
            recs={recs}
            onToggle={toggleItem}
          />

          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
            <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  2. Recommended For Your Basket (Top-10)
                </h3>
                <p className="text-xs text-slate-500">
                  Switch algorithms to compare how each model personalizes your storefront
                </p>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {(['SVD', 'Blend', 'ItemKNN', 'Popularity'] as LiveModel[]).map((m) => (
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
                ))}
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
              </div>
            )}

            {recs.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">
                Select at least one fashion product on the left (or click a Preset) to generate live recommendations.
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                {recs.map((r, idx) => {
                  const info = data.catalog[r.asin];
                  const pct = Math.max(6, Math.min(100, (Math.max(r.score, 0) / maxScore) * 100));
                  return (
                    <div
                      key={r.asin}
                      className="flex gap-3 rounded-xl border border-slate-200 bg-white p-3 hover:border-teal-400 hover:shadow-sm transition"
                    >
                      <div className="relative">
                        <ProductImage
                          src={info?.img}
                          alt={info?.t || r.asin}
                          category={info?.c || 'Fashion & Apparel'}
                          className="h-24 w-24"
                        />
                        <span className="absolute -top-2 -left-2 h-6 w-6 rounded-full bg-teal-700 text-white text-xs font-extrabold grid place-items-center shadow-xs">
                          {idx + 1}
                        </span>
                      </div>

                      <div className="min-w-0 flex-1 flex flex-col justify-between">
                        <div>
                          <a
                            href={asinUrl(r.asin)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs font-bold text-slate-900 hover:text-teal-700 line-clamp-2 leading-snug inline-flex items-start gap-1"
                          >
                            <span>{info?.t || r.asin}</span>
                            <ExternalLink className="h-3 w-3 shrink-0 mt-0.5 opacity-50" />
                          </a>
                          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-500">
                            <span className="rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-700">
                              {info?.c}
                            </span>
                            {info?.pr ? (
                              <span className="font-bold text-slate-900">${info.pr.toFixed(2)}</span>
                            ) : null}
                            <span>★ {info?.r.toFixed(1)} ({info?.n})</span>
                          </div>
                        </div>

                        <div className="mt-2 space-y-1.5">
                          <div className="inline-block rounded-md bg-teal-50 px-2 py-0.5 text-[10.5px] font-semibold text-teal-800 border border-teal-100">
                            {r.reason}
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                              <div
                                className="h-full rounded-full bg-teal-600"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <span className="text-[11px] font-bold text-slate-700 tabular-nums">
                              {r.score.toFixed(3)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
