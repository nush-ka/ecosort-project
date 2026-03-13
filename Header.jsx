import { PAGES } from '../utils/constants'

export default function Header({ page, onNav, mode, streaming, soundOn, onToggleSound, onReset }) {
  const modeColor = { yolov8: '#22c55e', mock: '#f59e0b', loading: '#4a7a4a', offline: '#ef4444' }[mode] || '#4a7a4a'
  const modeLabel = { yolov8: 'YOLOv8 AI', mock: 'Demo Mode', loading: 'Connecting…', offline: 'Offline' }[mode] || mode

  return (
    <header style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50,
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '16px 32px',
      borderBottom: '1px solid var(--border)',
      background: 'rgba(8,13,8,.92)',
      backdropFilter: 'blur(16px)',
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 36, height: 36, borderRadius: 9,
          background: 'linear-gradient(135deg, var(--green), var(--lime))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 17, boxShadow: '0 0 20px #22c55e44',
        }}>🌿</div>
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '1.2rem', color: 'var(--green)' }}>EcoSort</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '.6rem', color: 'var(--muted)', letterSpacing: '.12em', textTransform: 'uppercase' }}>AI Waste Classifier</div>
        </div>
      </div>

      {/* Nav links */}
      <nav style={{ display: 'flex', gap: 4 }}>
        {PAGES.map((name, i) => (
          <button key={name} onClick={() => onNav(i)} style={{
            fontFamily: 'var(--font-mono)', fontSize: '.72rem',
            padding: '6px 14px', borderRadius: 7, cursor: 'pointer',
            transition: 'all .2s', letterSpacing: '.05em', border: '1px solid',
            borderColor: page === i ? 'rgba(34,197,94,.3)' : 'transparent',
            color:       page === i ? 'var(--green)' : 'var(--muted)',
            background:  page === i ? 'rgba(34,197,94,.07)' : 'transparent',
          }}>{name}</button>
        ))}
      </nav>

      {/* Right controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {/* Mode badge */}
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '.66rem',
          padding: '4px 10px', borderRadius: 20,
          border: `1px solid ${modeColor}44`,
          color: modeColor, background: modeColor + '11',
        }}>{modeLabel}</div>

        {/* Live badge — only when streaming */}
        {streaming && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontFamily: 'var(--font-mono)', fontSize: '.68rem',
            padding: '5px 12px', borderRadius: 20,
            border: '1px solid rgba(34,197,94,.3)',
            color: 'var(--green)', background: 'rgba(34,197,94,.07)',
          }}>
            <span className="anim-pulse" style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)', display: 'inline-block' }} />
            LIVE
          </div>
        )}

        {/* Sound toggle */}
        <button onClick={onToggleSound} title={soundOn ? 'Mute sounds' : 'Enable sounds'} style={{
          width: 30, height: 30, borderRadius: 7, border: '1px solid var(--border)',
          background: 'transparent', color: 'var(--muted)', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1rem',
          transition: 'all .2s',
        }}>{soundOn ? '🔊' : '🔇'}</button>

        {/* Reset */}
        <button onClick={onReset} title="Reset session" style={{
          width: 30, height: 30, borderRadius: 7, border: '1px solid var(--border)',
          background: 'transparent', color: 'var(--muted)', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '.85rem',
          transition: 'all .2s',
        }}>↺</button>

        <button className="btn-primary" onClick={() => onNav(0)} style={{ padding: '8px 20px' }}>
          ▶ Start Camera
        </button>
      </div>
    </header>
  )
}
