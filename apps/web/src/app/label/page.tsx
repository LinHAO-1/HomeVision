import { notFound } from 'next/navigation';
import { LabelingTool } from '@/components/LabelingTool';

export default function LabelPage() {
  if (process.env.NEXT_PUBLIC_ENABLE_LABELING !== 'true') {
    notFound();
  }
  return (
    <main
      style={{
        minHeight: '100vh',
        width: '100%',
        padding: '2rem 1.5rem',
        boxSizing: 'border-box',
      }}
    >
      <h1 style={{ marginBottom: '0.5rem', fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
        Labeling Tool
      </h1>
      <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', fontSize: '1.05rem' }}>
        Upload a photo, review the model predictions, correct them, and save as ground truth for training.
      </p>
      <LabelingTool />
    </main>
  );
}
