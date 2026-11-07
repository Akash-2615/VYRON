export default function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand-mark ${compact ? "is-compact" : ""}`} aria-label="VYRON — Intelligence in Every Frame.">
      <div className="brand-row">
        <div className="brand-lockup">
          <img
            className="brand-logo"
            src="/brand/vyron-mark.png?v=1"
            alt=""
            width={compact ? 40 : 52}
            height={compact ? 36 : 48}
            decoding="async"
          />
        </div>
        <div className="brand-text">
          <span className="brand-name">VYRON</span>
          <small className="brand-aria">Powered by ARIA</small>
        </div>
      </div>
      {!compact && (
        <small className="brand-tagline">
          <span className="brand-tagline-text">Intelligence in Every Frame</span>
        </small>
      )}
    </div>
  );
}
