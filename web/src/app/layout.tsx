import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'StyleMatch — Fashion Recommendation Engine & Research Demo',
  description:
    'Interactive fashion product recommender comparing SVD, ItemKNN, Popularity, and Blend models on Amazon Clothing, Shoes & Jewelry.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-slate-50 text-slate-900 antialiased selection:bg-teal-600/20 selection:text-teal-900">
        {children}
      </body>
    </html>
  );
}
