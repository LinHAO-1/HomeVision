export default function Home() {
  return (
    <main
      style={{
        padding: '2rem 1.5rem',
        width: '100%',
        minHeight: '100vh',
      }}
    >
      <h1
        style={{
          marginBottom: '0.5rem',
          fontSize: '2rem',
          fontWeight: 700,
        }}
      >
        HomeVision
      </h1>
      <p
        style={{
          color: 'var(--text-muted)',
          fontSize: '1.05rem',
        }}
      >
        Upload real estate photos for room type, amenities, and photo quality
        analysis.
      </p>
    </main>
  );
}
