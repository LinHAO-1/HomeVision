import type { Metadata } from 'next';
import { Outfit } from 'next/font/google';
import './globals.css';

const outfit = Outfit({ subsets: ['latin'], variable: '--font-outfit' });

export const metadata: Metadata = {
  title: 'HomeVision',
  description: 'Real estate photo analysis',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={outfit.variable}>
      <body className={outfit.className}>
        <header
          style={{
            borderBottom: '1px solid var(--border)',
            padding: '0.75rem 1.5rem',
            background: 'var(--bg-card)',
          }}
        >
          <nav style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
            <a href="/" style={{ fontWeight: 700, fontSize: '1.25rem', color: 'var(--text)' }}>
              HomeVision
            </a>
            {process.env.NEXT_PUBLIC_ENABLE_LABELING === 'true' && (
              <a href="/label" style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Labeling Tool
              </a>
            )}
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
