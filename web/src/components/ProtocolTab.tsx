'use client';

import { ShieldCheck, Terminal } from 'lucide-react';
import { DemoData } from '@/lib/types';

export function ProtocolTab({ data }: { data: DemoData }) {
  const { meta } = data;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
          <h2 className="text-base font-bold text-slate-900">Chronological Evaluation Protocol</h2>
          <p className="text-xs text-slate-500 mb-4">
            Global time boundaries keep past and future strictly separated to prevent leakage
          </p>

          <div className="grid grid-cols-10 gap-1.5 mb-4">
            <div className="col-span-8 rounded-lg border border-teal-300 bg-teal-50 p-2.5">
              <div className="text-xs font-bold text-teal-900">Train (80%)</div>
              <div className="text-[11px] text-teal-700">
                {meta.date_range_utc[0].slice(0, 10)} → {meta.cutoffs_utc[0].slice(0, 10)} ·{' '}
                {meta.split_rows.train.toLocaleString()} reviews
              </div>
            </div>
            <div className="col-span-1 rounded-lg border border-amber-300 bg-amber-50 p-2 text-center">
              <div className="text-[11px] font-bold text-amber-900">Val</div>
              <div className="text-[10px] text-amber-700">10%</div>
            </div>
            <div className="col-span-1 rounded-lg border border-emerald-300 bg-emerald-50 p-2 text-center">
              <div className="text-[11px] font-bold text-emerald-900">Test</div>
              <div className="text-[10px] text-emerald-700">10%</div>
            </div>
          </div>

          <ul className="space-y-2 text-xs text-slate-700 list-disc pl-5 leading-relaxed">
            <li>
              <strong>Deduplication:</strong> Earliest review retained per unique user–product pair (
              {meta.unique_user_item_pairs.toLocaleString()} pairs).
            </li>
            <li>
              <strong>Implicit positives:</strong> Ratings &ge;4 train the positive interaction matrix (
              {meta.positive_training_events.toLocaleString()} events); lower ratings remain excluded as previously seen items.
            </li>
            <li>
              <strong>Frozen training window:</strong> All models are fitted strictly on training data; validation reviews are excluded as seen during test but never retrain models.
            </li>
            <li>
              <strong>Hyperparameters:</strong> ItemKNN = {meta.hyperparameters.item_knn_neighbors} cosine neighbors; TruncatedSVD = {meta.hyperparameters.svd_components} components, {meta.hyperparameters.svd_iterations} iterations, seed {meta.seed}; Blend &alpha; = {meta.selected_knn_weight}.
            </li>
          </ul>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
          <div className="flex items-center gap-2 mb-1">
            <ShieldCheck className="h-4 w-4 text-teal-700" />
            <h2 className="text-base font-bold text-slate-900">Verification &amp; Honest Scope</h2>
          </div>
          <p className="text-xs text-slate-500 mb-3">
            Automated checks enforced in scripts/benchmark.py &amp; scripts/verify_results.py
          </p>
          <div className="rounded-lg border-l-4 border-teal-600 bg-teal-50/70 p-3 text-xs text-slate-700 mb-4 leading-relaxed">
            {meta.verification}
          </div>
          <ul className="space-y-2 text-xs text-slate-700 list-disc pl-5 leading-relaxed">
            <li>
              <strong>Official product metadata:</strong> Product titles, prices, and images are matched from the official Amazon 2014 Clothing, Shoes &amp; Jewelry metadata archive ({data.metadata_stats?.with_official_images ?? 0}/{data.metadata_stats?.total_catalog ?? 0} demo products matched).
            </li>
            <li>
              <strong>Warm-start scope:</strong> Evaluates users with &ge;3 training reviews and held-out products present in the training-positive catalog.
            </li>
            <li>
              <strong>Privacy &amp; integrity:</strong> Raw reviewer IDs and review text are never exported; users appear as numeric indexes.
            </li>
          </ul>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
        <div className="flex items-center gap-2 mb-1">
          <Terminal className="h-4 w-4 text-teal-700" />
          <h2 className="text-base font-bold text-slate-900">Reproduce Locally</h2>
        </div>
        <p className="text-xs text-slate-500 mb-3">
          Run the offline benchmark, verification checks, metadata export, and Next.js static site build
        </p>
        <pre className="rounded-lg bg-slate-900 text-slate-100 p-4 text-xs font-mono overflow-x-auto leading-relaxed">
{`git clone https://github.com/FaisalKabirGalib/StyleMatch.git
cd StyleMatch
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/verify_results.py
.venv/bin/python scripts/build_demo_data.py
cd web && npm install && npm run build`}
        </pre>
        <p className="text-xs text-slate-500 mt-3">
          Measured environment: Python {meta.environment.python}, NumPy {meta.environment.numpy}, Pandas{' '}
          {meta.environment.pandas}, SciPy {meta.environment.scipy}, scikit-learn {meta.environment.sklearn} &middot; Benchmark runtime: {meta.elapsed_seconds.toFixed(1)}s.
        </p>
      </div>
    </div>
  );
}
