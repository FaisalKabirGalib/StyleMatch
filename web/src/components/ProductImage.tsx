'use client';

import { useState } from 'react';
import {
  Watch,
  Footprints,
  Gem,
  ShoppingBag,
  Glasses,
  Shirt,
  Sparkles,
} from 'lucide-react';
import { CATEGORY_COLORS } from '@/lib/types';

function CategoryIcon({ category, className }: { category: string; className?: string }) {
  if (category === 'Watches') return <Watch className={className} />;
  if (category.includes('Sneakers') || category.includes('Boots') || category.includes('Sandals'))
    return <Footprints className={className} />;
  if (category.includes('Jewelry') || category.includes('Rings')) return <Gem className={className} />;
  if (category.includes('Bags')) return <ShoppingBag className={className} />;
  if (category.includes('Sunglasses')) return <Glasses className={className} />;
  if (category.includes('Shirts') || category.includes('Jackets') || category.includes('Dresses') || category.includes('Pants'))
    return <Shirt className={className} />;
  return <Sparkles className={className} />;
}

export function ProductImage({
  src,
  alt,
  category,
  className = 'h-24 w-24',
}: {
  src?: string;
  alt: string;
  category: string;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  const accent = CATEGORY_COLORS[category] || '#0f766e';

  if (src && !failed) {
    return (
      <div
        className={`relative shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-white flex items-center justify-center p-1.5 ${className}`}
      >
        <img
          src={src}
          alt={alt}
          loading="lazy"
          referrerPolicy="no-referrer"
          onError={() => setFailed(true)}
          className="max-h-full max-w-full object-contain"
        />
      </div>
    );
  }

  return (
    <div
      className={`relative shrink-0 overflow-hidden rounded-lg border border-slate-200 flex flex-col items-center justify-center gap-1 p-2 ${className}`}
      style={{
        background: `linear-gradient(135deg, ${accent}14 0%, ${accent}28 100%)`,
        color: accent,
      }}
    >
      <CategoryIcon category={category} className="h-6 w-6 opacity-85" />
      <span className="text-[10px] font-semibold text-center leading-tight line-clamp-1 opacity-80">
        {category.split(' ')[0]}
      </span>
    </div>
  );
}
