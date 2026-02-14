'use client';

import { useState, useCallback, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
const INFERENCE_URL = process.env.NEXT_PUBLIC_INFERENCE_URL || 'http://localhost:8000';

const ALL_ROOM_TYPES = [
  'Kitchen', 'Bathroom', 'Bedroom', 'Living Room', 'Dining Room', 'Exterior', 'Unknown',
];

const ALL_AMENITIES = [
  'Stainless Steel Appliances', 'Fireplace', 'Pool',
  'View', 'Natural Light', 'Updated Kitchen',
];

const COUNTERTOP_FEATURES = [
  'Granite Countertop', 'Marble Countertop', 'Quartz Countertop', 'Stone Countertop', 'Stainless Steel Countertop', 'Wood Countertop',
];

function featuresWithSingleCountertop(featureList: string[]): Set<string> {
  const set = new Set(featureList);
  const countertopsInSet = COUNTERTOP_FEATURES.filter((c) => set.has(c));
  if (countertopsInSet.length <= 1) return set;
  countertopsInSet.slice(1).forEach((c) => set.delete(c));
  return set;
}

const ALL_FEATURES: Record<string, string[]> = {
  Kitchen: [
    'Kitchen Island', ...COUNTERTOP_FEATURES,
    'Gas Stove', 'Electric Stove', 'Tile Backsplash', 'Pantry', 'Double Oven',
    'Breakfast Bar', 'Under-Cabinet Lighting', 'Dishwasher',
  ],
  Bathroom: [
    'Walk-In Shower', 'Bathtub', 'Double Vanity', 'Tile Floor',
    'Soaking Tub', 'Glass Shower Door',
  ],
  'Living Space': [
    'Crown Molding', 'Recessed Lighting', 'Hanging Lights', 'Ceiling Fan',
    'Built-In Shelving', 'Wainscoting', 'Exposed Brick', 'Accent Wall',
  ],
  Bedroom: ['Walk-In Closet', 'En-Suite Bathroom', 'Bay Window', 'Window Seat'],
  Flooring: ['Hardwood Floors', 'Carpet', 'Tile Flooring', 'Laminate Flooring', 'Stone Flooring'],
  Exterior: [
    'Patio', 'Deck', 'Garage', 'Front Porch', 'Landscaping',
    'Fenced Yard', 'Driveway', 'Outdoor Kitchen', 'Pergola', 'Garden',
  ],
  General: [
    'Open Floor Plan', 'Vaulted Ceiling', 'Washer/Dryer', 'Central AC Unit',
    'Skylight', 'French Doors', 'Sliding Glass Door', 'Staircase',
    'Laundry Room', 'Home Office', 'Storage Space',
  ],
};

type InferenceResult = {
  filename: string;
  roomType: { label: string };
  amenities: Array<{ label: string }>;
  features: Array<{ label: string; category: string }>;
};

type SavedLabel = {
  id: number;
  filename: string;
  roomType: string;
  amenities: string[];
  features: string[];
};

export function LabelingTool() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Editable label state
  const [roomType, setRoomType] = useState('Unknown');
  const [amenities, setAmenities] = useState<Set<string>>(new Set());
  const [features, setFeatures] = useState<Set<string>>(new Set());
  const [predictions, setPredictions] = useState<InferenceResult | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [existingLabelLoaded, setExistingLabelLoaded] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setSaved(false);
    setError(null);
    setPredictions(null);
    setExistingLabelLoaded(false);
    setSaveMessage(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  // When a file is selected, check if we already have a label for this filename and pre-fill
  useEffect(() => {
    if (!file?.name) return;
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch(
          `${API_BASE}/api/v1/labels?filename=${encodeURIComponent(file.name)}`
        );
        if (!res.ok || cancelled) return;
        const data: SavedLabel[] = await res.json();
        if (cancelled) return;
        if (data.length > 0) {
          const label = data[0];
          setRoomType(label.roomType);
          const amenities = label.amenities.filter((a) => a !== 'Hardwood Floors');
          setAmenities(new Set(amenities));
          const features = [...label.features];
          if (label.amenities.includes('Hardwood Floors') && !features.includes('Hardwood Floors'))
            features.push('Hardwood Floors');
          setFeatures(featuresWithSingleCountertop(features));
          setExistingLabelLoaded(true);
        } else {
          setRoomType('Unknown');
          setAmenities(new Set());
          setFeatures(new Set());
        }
      } catch {
        if (!cancelled) {
          setRoomType('Unknown');
          setAmenities(new Set());
          setFeatures(new Set());
        }
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [file?.name]);

  const analyzePhoto = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setSaved(false);
    try {
      const form = new FormData();
      form.append('files', file);
      const res = await fetch(`${INFERENCE_URL}/analyze/batch`, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) {
        setError('Inference failed');
        setLoading(false);
        return;
      }
      const data: InferenceResult[] = await res.json();
      const result = data[0];
      setPredictions(result);
      setExistingLabelLoaded(false);

      // Pre-fill labels from model predictions
      setRoomType(result.roomType.label);
      setAmenities(new Set(result.amenities.map((a) => a.label)));
      setFeatures(featuresWithSingleCountertop(result.features.map((f) => f.label)));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error');
    }
    setLoading(false);
  }, [file]);

  const toggleAmenity = (label: string) => {
    setAmenities((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  };

  const toggleFeature = (label: string) => {
    setFeatures((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        if (COUNTERTOP_FEATURES.includes(label)) {
          COUNTERTOP_FEATURES.forEach((c) => next.delete(c));
        }
        next.add(label);
      }
      return next;
    });
  };

  const resetForNext = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setRoomType('Unknown');
    setAmenities(new Set());
    setFeatures(new Set());
    setPredictions(null);
    setSaved(false);
    setError(null);
    setExistingLabelLoaded(false);
    setFileInputKey((k) => k + 1);
  }, [preview]);

  const saveLabel = useCallback(async () => {
    if (!file) return;
    setError(null);
    setSaveMessage(null);
    try {
      const body = {
        filename: file.name,
        roomType,
        amenities: Array.from(amenities),
        features: Array.from(features),
        modelPredictions: predictions ?? undefined,
      };
      const res = await fetch(`${API_BASE}/api/v1/labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        setError('Failed to save label');
        return;
      }
      const savedLabel: SavedLabel = await res.json();
      const msg = `Saved! Label #${savedLabel.id} stored. Ready for next photo.`;
      resetForNext();
      setSaveMessage(msg);
      setTimeout(() => setSaveMessage(null), 1800);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save error');
    }
  }, [file, roomType, amenities, features, predictions, resetForNext]);

  const chipStyle = (active: boolean, color?: string) => ({
    padding: '4px 10px',
    borderRadius: 4,
    fontSize: '0.8rem',
    cursor: 'pointer',
    border: active ? '2px solid #3b82f6' : '1px solid #475569',
    background: active ? (color || '#1e3a5f') : '#1e293b',
    color: active ? '#93c5fd' : '#94a3b8',
    userSelect: 'none' as const,
  });

  return (
    <div style={{ minHeight: 'calc(100vh - 8rem)', display: 'flex', flexDirection: 'column' }}>
      {/* File picker + analyze */}
      <div style={{ marginBottom: '1rem', display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
        <input key={fileInputKey} type="file" accept="image/*" onChange={onFileChange} />
        <button
          onClick={analyzePhoto}
          disabled={!file || loading}
          style={{
            padding: '0.5rem 1rem',
            background: '#3b82f6',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            cursor: file && !loading ? 'pointer' : 'not-allowed',
          }}
        >
          {loading ? 'Analyzing…' : 'Analyze & Pre-fill'}
        </button>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', marginBottom: '1rem', background: 'rgba(239,68,68,0.2)', borderRadius: 6, color: '#fca5a5' }}>
          {error}
        </div>
      )}

      {saveMessage && (
        <div style={{ padding: '0.75rem', marginBottom: '1rem', background: 'rgba(34,197,94,0.2)', borderRadius: 6, color: '#86efac' }}>
          {saveMessage}
        </div>
      )}

      {existingLabelLoaded && preview && (
        <div style={{ padding: '0.5rem 0.75rem', marginBottom: '1rem', background: 'rgba(59,130,246,0.2)', borderRadius: 6, color: '#93c5fd', fontSize: '0.9rem' }}>
          Loaded existing label for this photo. You can edit and save again to update.
        </div>
      )}

      {preview && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(55%, 1.4fr) 1fr',
          gap: '1.5rem',
          flex: 1,
          minHeight: 0,
          alignItems: 'start',
        }}>
          {/* Left: image */}
          <div style={{ minWidth: 0, position: 'sticky', top: '1rem' }}>
            <img
              src={preview}
              alt="Preview"
              style={{
                width: '100%',
                maxHeight: 'calc(100vh - 6rem)',
                minHeight: 360,
                objectFit: 'contain',
                borderRadius: 8,
                border: '1px solid #334155',
              }}
            />
          </div>

          {/* Right: label editor */}
          <div style={{ minWidth: 0, overflowY: 'auto' }}>
            <div style={{ marginBottom: '1rem' }}>
              <p style={{ fontWeight: 600, marginBottom: 4 }}>Room Type</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {ALL_ROOM_TYPES.map((r) => (
                  <span
                    key={r}
                    onClick={() => setRoomType(r)}
                    style={chipStyle(roomType === r)}
                  >
                    {r}
                  </span>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <p style={{ fontWeight: 600, marginBottom: 4 }}>Amenities</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {ALL_AMENITIES.map((a) => (
                  <span
                    key={a}
                    onClick={() => toggleAmenity(a)}
                    style={chipStyle(amenities.has(a))}
                  >
                    {a}
                  </span>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <p style={{ fontWeight: 600, marginBottom: 4 }}>Features</p>
              {Object.entries(ALL_FEATURES).map(([cat, labels]) => (
                <div key={cat} style={{ marginBottom: 8 }}>
                  <p style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: 2 }}>{cat}</p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {labels.map((f) => (
                      <span
                        key={f}
                        onClick={() => toggleFeature(f)}
                        style={chipStyle(features.has(f))}
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={saveLabel}
              style={{
                padding: '0.5rem 1.5rem',
                background: saved ? '#065f46' : '#16a34a',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: '1rem',
              }}
            >
              {saved ? 'Saved!' : 'Save Label'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
