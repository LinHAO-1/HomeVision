import type { Metadata } from 'next';
import './globals.css';

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
    <html lang="en">
      <body>
        <header
          style={{
            borderBottom: '1px solid var(--border)',
            padding: '0.75rem 1.5rem',
            background: 'var(--bg-card)',
          }}
        >
          <nav style={{ display: 'flex', alignItems: 'center' }}>
            <a
              href="/"
              style={{
                fontWeight: 700,
                fontSize: '1.25rem',
                color: 'var(--text)',
              }}
            >
              HomeVision
            </a>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
