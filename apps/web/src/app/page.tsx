import { HomeVisionUpload } from '@/components/HomeVisionUpload';

export default function Home() {
  return (
    <main style={{ padding: '2rem 1.5rem', width: '100%', minHeight: '100vh' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ marginBottom: '0.5rem', fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
          HomeVision
        </h1>
        <p style={{ color: 'var(--text-muted)', marginBottom: '0.75rem', fontSize: '1.05rem' }}>
          Upload 1–20 real estate photos for room type, amenities, and photo quality analysis.
        </p>
        <div
          style={{
            display: 'inline-block',
            padding: '0.35rem 0.75rem',
            background: 'var(--accent-muted)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.8125rem',
            color: 'var(--text-muted)',
          }}
        >
          Optimized for <strong style={{ color: 'var(--accent)' }}>Kitchen</strong> · Tuning: Bathroom, Bedroom, Living Room, Dining Room, Exterior
        </div>
      </div>
      <HomeVisionUpload />
    </main>
  );
}
