'use client';

import { useState, useCallback, useEffect, useRef } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

/** Labels that represent room type; adapter may predict these. We show one as "Room" and omit from Features. */
const ROOM_TYPE_LABELS = new Set([
  'Kitchen', 'Bathroom', 'Bedroom', 'Living Room', 'Dining Room', 'Exterior',
]);

type JobStatus = 'processing' | 'completed' | 'failed';

type TopAmenity = { label: string; count: number; avgScore: number };
type TopFeature = { label: string; category: string; count: number; avgScore: number };
type PhotoResult = {
  filename: string;
  roomType: { label: string; score: number; topPrompt: string };
  amenities: Array<{ label: string; score: number; prompt: string }>;
  features: Array<{ label: string; score: number; category: string; prompt: string }>;
  quality: {
    blurVar: number;
    brightness: number;
    width: number;
    height: number;
    isBlurry: boolean;
    isDark: boolean;
    overallScore: number;
  };
  adapterPredictions?: Array<{ label: string; confidence: number }>;
};

/** Derive room display and feature list: use adapter room when available and don't duplicate in Features. */
function getRoomAndFeatures(photo: PhotoResult): {
  roomLabel: string;
  roomScore: number;
  featurePredictions: Array<{ label: string; confidence: number }>;
} {
  const adapter = photo.adapterPredictions ?? [];
  const roomFromAdapter = adapter
    .filter((p) => ROOM_TYPE_LABELS.has(p.label))
    .sort((a, b) => b.confidence - a.confidence)[0];
  if (roomFromAdapter) {
    return {
      roomLabel: roomFromAdapter.label,
      roomScore: roomFromAdapter.confidence,
      featurePredictions: adapter.filter((p) => p.label !== roomFromAdapter.label),
    };
  }
  return {
    roomLabel: photo.roomType.label,
    roomScore: photo.roomType.score,
    featurePredictions: adapter,
  };
}

type JobResponse = {
  id: string;
  status: JobStatus;
  results: null | {
    summary: {
      topAmenities: TopAmenity[];
      topFeatures: TopFeature[];
      overallQualityScore: number;
    };
    photos: PhotoResult[];
  };
  errorMessage: string | null;
};

const MAX_FILES = 20;

export function HomeVisionUpload() {
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [photoUrls, setPhotoUrls] = useState<Record<string, string>>({});
  const photoUrlsRef = useRef<Record<string, string>>({});
  const [selectedPhoto, setSelectedPhoto] = useState<PhotoResult | null>(null);

  // Create object URLs for selected files when we have results, so we can show the analyzed photos
  useEffect(() => {
    if (!job?.results?.photos?.length || !files.length) return;
    Object.values(photoUrlsRef.current).forEach((u) => URL.revokeObjectURL(u));
    const urls: Record<string, string> = {};
    const fileMap = new Map<string, File>();
    for (let i = 0; i < files.length; i++) {
      fileMap.set(files[i].name, files[i]);
    }
    job.results.photos.forEach((photo) => {
      const file = fileMap.get(photo.filename);
      if (file) urls[photo.filename] = URL.createObjectURL(file);
    });
    photoUrlsRef.current = urls;
    setPhotoUrls(urls);
    return () => {
      Object.values(photoUrlsRef.current).forEach((u) => URL.revokeObjectURL(u));
      photoUrlsRef.current = {};
    };
  }, [job?.results, files]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedPhoto(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files;
    if (!list?.length) return;
    const newFiles = Array.from(list);
    setFiles((prev) => {
      const byName = new Map<string, File>();
      prev.forEach((f) => byName.set(f.name, f));
      newFiles.forEach((f) => byName.set(f.name, f));
      const merged = Array.from(byName.values());
      const capped = merged.length > MAX_FILES ? merged.slice(0, MAX_FILES) : merged;
      if (merged.length > MAX_FILES) {
        setTimeout(() => setError(`Maximum ${MAX_FILES} files. Keeping first ${MAX_FILES}.`), 0);
      } else {
        setTimeout(() => setError(null), 0);
      }
      return capped;
    });
    e.target.value = '';
  }, []);

  const clearAll = useCallback(() => {
    setFiles([]);
    setError(null);
    setJobId(null);
    setJob(null);
    setSelectedPhoto(null);
  }, []);

  const submit = useCallback(async () => {
    if (!files.length) {
      setError('Select 1-20 photos to upload.');
      return;
    }
    setError(null);
    setUploading(true);
    try {
      const form = new FormData();
      for (let i = 0; i < files.length; i++) {
        form.append('files', files[i]);
      }
      const res = await fetch(`${API_BASE}/api/v1/jobs`, {
        method: 'POST',
        body: form,
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data?.message || res.statusText || 'Upload failed');
        setUploading(false);
        return;
      }
      setJobId(data.jobId);
      setJob({ id: data.jobId, status: 'processing', results: null, errorMessage: null });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Network error');
    }
    setUploading(false);
  }, [files]);

  const pollJob = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/jobs/${id}`);
      const data: JobResponse = await res.json();
      if (!res.ok) {
        setError(data?.errorMessage ?? 'Failed to fetch job');
        return;
      }
      setJob(data);
      if (data.status === 'processing') {
        setTimeout(() => pollJob(id), 1000);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Poll error');
    }
  }, []);

  useEffect(() => {
    if (!jobId || !job || job.status !== 'processing' || job.results) return;
    const t = setTimeout(() => pollJob(jobId), 1000);
    return () => clearTimeout(t);
  }, [jobId, job?.status, job?.results, pollJob]);

  return (
    <div>
      {/* Upload controls */}
      <div
        style={{
          marginBottom: '1.5rem',
          padding: '1.5rem',
          background: 'var(--bg-card)',
          borderRadius: 'var(--radius)',
          border: '1px solid var(--border)',
        }}
      >
        <p style={{ marginBottom: '0.75rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Upload 1–20 photos (image/*, max 5MB each).
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem' }}>
          <label
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              padding: '0.5rem 1rem',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              cursor: uploading ? 'not-allowed' : 'pointer',
              fontSize: '0.9rem',
              color: 'var(--text-muted)',
            }}
          >
            Choose files
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={onFileChange}
              disabled={uploading}
              style={{ display: 'none' }}
            />
          </label>
          <button
            type="button"
            onClick={submit}
            disabled={!files.length || uploading}
            style={{
              padding: '0.5rem 1.25rem',
              background: files.length && !uploading ? 'var(--accent)' : 'var(--bg-elevated)',
              color: files.length && !uploading ? '#0c1222' : 'var(--text-dim)',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              cursor: files.length && !uploading ? 'pointer' : 'not-allowed',
              fontWeight: 600,
              fontSize: '0.9rem',
            }}
          >
            {uploading ? 'Uploading…' : 'Analyze'}
          </button>
          {files.length > 0 && (
            <>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                {files.length} file{files.length !== 1 ? 's' : ''} selected
              </span>
              <button
                type="button"
                onClick={clearAll}
                style={{
                  padding: '0.5rem 1rem',
                  background: 'transparent',
                  color: 'var(--text-muted)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  fontSize: '0.875rem',
                }}
              >
                Clear all
              </button>
            </>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            padding: '0.75rem 1rem',
            marginBottom: '1rem',
            background: 'var(--error-bg)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--error-text)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
          }}
        >
          {error}
        </div>
      )}

      {/* Hint when more files selected than in current results */}
      {job?.results && files.length > job.results.photos.length && (
        <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.875rem' }}>
          {files.length - job.results.photos.length} more photo(s) selected. Click Analyze to include them.
        </p>
      )}

      {/* Processing */}
      {job?.status === 'processing' && !job.results && (
        <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>Processing…</p>
      )}

      {/* Failed */}
      {job?.status === 'failed' && job.errorMessage && (
        <div
          style={{
            padding: '0.75rem 1rem',
            marginBottom: '1rem',
            background: 'var(--error-bg)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--error-text)',
          }}
        >
          Job failed: {job.errorMessage}
        </div>
      )}

      {/* Results */}
      {job?.results && (
        <div style={{ marginTop: '2rem' }}>
          {/* Legend */}
          <details style={{ marginBottom: '1.5rem' }}>
            <summary
              style={{
                cursor: 'pointer',
                color: 'var(--text-muted)',
                fontSize: '0.875rem',
              }}
            >
              What do these scores mean?
            </summary>
            <div
              style={{
                marginTop: '0.5rem',
                padding: '1rem 1.25rem',
                background: 'var(--bg-card)',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--border)',
                fontSize: '0.8125rem',
                color: 'var(--text-muted)',
                lineHeight: 1.6,
              }}
            >
              <p style={{ margin: '0 0 0.5rem 0' }}>
                <strong style={{ color: 'var(--text)' }}>Room (score)</strong> — Primary room type; score is model confidence from 0 to 1.
              </p>
              <p style={{ margin: '0 0 0.5rem 0' }}>
                <strong style={{ color: 'var(--text)' }}>Features (scores)</strong> — Confidence (0–1) that each feature is present. Higher = more confident.
              </p>
              <p style={{ margin: 0 }}>
                <strong style={{ color: 'var(--text)' }}>Photo quality (0–1)</strong> — Sharpness, brightness, and size. 1 = best.
              </p>
            </div>
          </details>

          {/* Photo modal */}
          {selectedPhoto && photoUrls[selectedPhoto.filename] && (
            <div
              role="dialog"
              aria-modal="true"
              aria-label={`Photo details: ${selectedPhoto.filename}`}
              onClick={() => setSelectedPhoto(null)}
              style={{
                position: 'fixed',
                inset: 0,
                background: 'rgba(0,0,0,0.75)',
                backdropFilter: 'blur(4px)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 50,
                padding: '2rem',
              }}
            >
              <div
                onClick={(e) => e.stopPropagation()}
                style={{
                  background: 'var(--bg-card)',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border)',
                  boxShadow: 'var(--shadow)',
                  maxWidth: '90vw',
                  maxHeight: '90vh',
                  overflow: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '0.5rem 0.75rem 0' }}>
                  <button
                    type="button"
                    onClick={() => setSelectedPhoto(null)}
                    aria-label="Close"
                    style={{
                      background: 'var(--bg-elevated)',
                      border: 'none',
                      color: 'var(--text-muted)',
                      width: 32,
                      height: 32,
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '1.25rem',
                      cursor: 'pointer',
                      lineHeight: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    ×
                  </button>
                </div>
                <img
                  src={photoUrls[selectedPhoto.filename]}
                  alt={selectedPhoto.filename}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '70vh',
                    objectFit: 'contain',
                    padding: '0 1.5rem 1rem',
                  }}
                />
                <div style={{ padding: '0 1.5rem 1.5rem' }}>
                  <p style={{ fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text)' }}>{selectedPhoto.filename}</p>
                  {(() => {
                    const { roomLabel, roomScore, featurePredictions } = getRoomAndFeatures(selectedPhoto);
                    return (
                      <>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                          Room: {roomLabel} (score: {roomScore})
                        </p>
                        {selectedPhoto.amenities.length > 0 && (
                          <div style={{ marginTop: '0.5rem' }}>
                            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 4 }}>Amenities</p>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                              {selectedPhoto.amenities.map((a) => (
                                <span
                                  key={a.label}
                                  style={{
                                    padding: '4px 8px',
                                    background: 'var(--bg-elevated)',
                                    borderRadius: 'var(--radius-sm)',
                                    fontSize: '0.75rem',
                                    border: '1px solid var(--border)',
                                  }}
                                >
                                  {a.label} ({a.score})
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        {featurePredictions.length > 0 && (
                          <div style={{ marginTop: '0.5rem' }}>
                            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 4 }}>Features</p>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                              {featurePredictions.map((p) => (
                                <span
                                  key={p.label}
                                  style={{
                                    padding: '4px 8px',
                                    background: 'var(--feature-bg)',
                                    color: '#f3f0ff',
                                    borderRadius: 'var(--radius-sm)',
                                    fontSize: '0.75rem',
                                  }}
                                >
                                  {p.label} ({p.confidence})
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                          Photo quality: {selectedPhoto.quality.overallScore}
                          {selectedPhoto.quality.isBlurry && ' · Blurry'}
                          {selectedPhoto.quality.isDark && ' · Dark'}
                          {' · '}
                          {selectedPhoto.quality.width}×{selectedPhoto.quality.height}
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>
            </div>
          )}

          {/* Summary */}
          <h2 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text)' }}>Summary</h2>
          <div
            style={{
              padding: '1.25rem',
              background: 'var(--bg-card)',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--border)',
              marginBottom: '1.5rem',
            }}
          >
            <p style={{ color: 'var(--text-muted)' }}>
              <strong style={{ color: 'var(--text)' }}>Overall photo quality score:</strong>{' '}
              {job.results.summary.overallQualityScore}
            </p>

            {job.results.summary.topAmenities?.length > 0 && (
              <div style={{ marginTop: '0.75rem' }}>
                <strong style={{ color: 'var(--text)' }}>Top amenities:</strong>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: '0.35rem' }}>
                  {job.results.summary.topAmenities.map((a) => (
                    <span
                      key={a.label}
                      style={{
                        padding: '0.25rem 0.5rem',
                        background: 'var(--bg-elevated)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.875rem',
                        border: '1px solid var(--border)',
                      }}
                    >
                      {a.label} (×{a.count}, avg {a.avgScore})
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Photo grid */}
          <h2 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text)' }}>Photos</h2>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
              gap: '1rem',
            }}
          >
            {job.results.photos.map((photo) => (
              <div
                key={photo.filename}
                role="button"
                tabIndex={0}
                onClick={() => setSelectedPhoto(photo)}
                onKeyDown={(e) => e.key === 'Enter' && setSelectedPhoto(photo)}
                style={{
                  padding: '1rem',
                  background: 'var(--bg-card)',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border)',
                  cursor: 'pointer',
                  transition: 'border-color 0.15s, box-shadow 0.15s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-focus)';
                  e.currentTarget.style.boxShadow = 'var(--shadow)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                {photoUrls[photo.filename] && (
                  <img
                    src={photoUrls[photo.filename]}
                    alt={photo.filename}
                    style={{
                      width: '100%',
                      maxHeight: 220,
                      objectFit: 'contain',
                      borderRadius: 'var(--radius-sm)',
                      marginBottom: '0.75rem',
                      background: 'var(--bg)',
                    }}
                  />
                )}
                <p style={{ fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text)', fontSize: '0.9rem' }}>{photo.filename}</p>
                {(() => {
                  const { roomLabel, roomScore, featurePredictions } = getRoomAndFeatures(photo);
                  return (
                    <>
                      <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                        Room: {roomLabel} (score: {roomScore})
                      </p>
                      {photo.amenities.length > 0 && (
                        <div style={{ marginTop: '0.5rem' }}>
                          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 4 }}>Amenities</p>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {photo.amenities.map((a) => (
                              <span
                                key={a.label}
                                style={{
                                  padding: '4px 8px',
                                  background: 'var(--bg-elevated)',
                                  borderRadius: 'var(--radius-sm)',
                                  fontSize: '0.75rem',
                                  border: '1px solid var(--border)',
                                }}
                              >
                                {a.label} ({a.score})
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {featurePredictions.length > 0 && (
                        <div style={{ marginTop: '0.5rem' }}>
                          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 4 }}>Features</p>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {featurePredictions.map((p) => (
                              <span
                                key={p.label}
                                style={{
                                  padding: '4px 8px',
                                  background: 'var(--feature-bg)',
                                  color: '#f3f0ff',
                                  borderRadius: 'var(--radius-sm)',
                                  fontSize: '0.75rem',
                                }}
                              >
                                {p.label} ({p.confidence})
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                        Photo quality: {photo.quality.overallScore}
                        {photo.quality.isBlurry && ' · Blurry'}
                        {photo.quality.isDark && ' · Dark'}
                        {' · '}
                        {photo.quality.width}×{photo.quality.height}
                      </div>
                    </>
                  );
                })()}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
