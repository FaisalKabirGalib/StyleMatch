'use client';

import { useEffect, useState } from 'react';
import {
  BarChart3,
  BookOpen,
  ExternalLink,
  FileText,
  ShoppingBag,
  Sparkles,
  Users,
} from 'lucide-react';
import { DemoData } from '@/lib/types';
import { OverviewTab } from '@/components/OverviewTab';
import { UserExplorerTab } from '@/components/UserExplorerTab';
import { BasketStudioTab } from '@/components/BasketStudioTab';
import { ProtocolTab } from '@/components/ProtocolTab';

type TabKey = 'studio' | 'explorer' | 'overview' | 'protocol';

const TABS: { key: TabKey; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { key: 'studio', label: 'Live Recommender Studio', icon: ShoppingBag },
  { key: 'explorer', label: 'Test User Explorer', icon: Users },
  { key: 'overview', label: 'Benchmark & Charts', icon: BarChart3 },
  { key: 'protocol', label: 'Protocol & Code', icon: BookOpen },
];

export default function HomePage() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>('studio');

  useEffect(() => {
    fetch('data/demo_data.json')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: DemoData) => setData(json))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-teal-700 text-white grid place-items-center font-black text-sm shadow-xs">
              SM
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-900">
                  StyleMatch
                </h1>
                <span className="hidden sm:inline-flex items-center gap-1 rounded-full bg-teal-50 border border-teal-200 px-2 py-0.5 text-[10.5px] font-bold text-teal-800">
                  <Sparkles className="h-2.5 w-2.5" /> Fashion RecSys Demo
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Collaborative Filtering &amp; Matrix Factorization on Amazon Fashion
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <a
              href="progress_report.html"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
            >
              <FileText className="h-3.5 w-3.5 text-teal-700" />
              Progress Report
            </a>
            <a
              href="https://github.com/FaisalKabirGalib/StyleMatch"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800"
            >
              GitHub <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="border-t border-slate-100 bg-white">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 flex gap-1 overflow-x-auto">
            {TABS.map(({ key, label, icon: Icon }) => {
              const active = tab === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setTab(key)}
                  className={`inline-flex items-center gap-2 px-3.5 py-3 text-xs sm:text-sm font-semibold border-b-2 transition whitespace-nowrap cursor-pointer ${
                    active
                      ? 'border-teal-700 text-teal-800'
                      : 'border-transparent text-slate-500 hover:text-slate-800'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </button>
              );
            })}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 sm:px-6 py-6">
        {error ? (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800">
            <strong>Failed to load demo data:</strong> {error}
          </div>
        ) : !data ? (
          <div className="py-24 text-center text-sm text-slate-500">
            Loading trained StyleMatch models &amp; fashion catalog…
          </div>
        ) : (
          <>
            {tab === 'studio' && <BasketStudioTab data={data} />}
            {tab === 'explorer' && <UserExplorerTab data={data} />}
            {tab === 'overview' && <OverviewTab data={data} />}
            {tab === 'protocol' && <ProtocolTab data={data} />}
          </>
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white py-4">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
          <span>StyleMatch · Amazon 2014 Clothing, Shoes &amp; Jewelry 5-Core Research Pilot</span>
          <span>Built with Next.js Static Export · Hosted on GitHub Pages</span>
        </div>
      </footer>
    </div>
  );
}
