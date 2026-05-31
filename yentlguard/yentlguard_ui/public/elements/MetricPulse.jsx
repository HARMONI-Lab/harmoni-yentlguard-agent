export default function MetricPulse() {
  const metrics = (props && props.metrics) || {};

  const SPECS = {
    delta_m: { name: "ΔM",  full: "Confidence margin shift", min: 0, max: 3,
      bands: [[0, 0.5, "good"], [0.5, 1.5, "warn"], [1.5, 3, "alert"]] },
    crr:     { name: "CRR", full: "Confidence recovery",     min: 0, max: 1,
      bands: [[0, 0.5, "alert"], [0.5, 0.8, "warn"], [0.8, 1, "good"]] },
    tar:     { name: "TAR", full: "Thought allocation",      min: 0, max: 2,
      bands: [[0, 0.8, "warn"], [0.8, 1.2, "good"], [1.2, 2, "warn"]] },
    gap:     { name: "GAP", full: "Sycophancy gap",          min: 0, max: 1,
      bands: [[0, 0.15, "good"], [0.15, 0.35, "warn"], [0.35, 1, "alert"]] },
    pss:     { name: "PSS", full: "Prompt sensitivity",      min: 0, max: 1,
      bands: [[0, 0.33, "good"], [0.33, 0.66, "warn"], [0.66, 1, "alert"]] },
  };
  const COLORS = { good: "#1D9E75", warn: "#E0A33E", alert: "#D85A30" };

  const classify = (spec, v) => {
    for (const band of spec.bands) if (v >= band[0] && v < band[1]) return band[2];
    return "warn";
  };

  const Gauge = ({ spec, value }) => {
    const v = parseFloat(value);
    const frac = Math.max(0, Math.min(1, (v - spec.min) / (spec.max - spec.min)));
    const cls = classify(spec, v);
    const color = COLORS[cls] || "#8b949e";
    const R = 26, C = 2 * Math.PI * R, dash = C * 0.75; // 270-degree arc
    const off = dash * (1 - frac);
    const cell = { display: "flex", flexDirection: "column", alignItems: "center", minWidth: 96, padding: "10px 6px" };
    const arc = { transition: "stroke-dashoffset .8s ease, stroke .4s" };
    const nameStyle = { fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontWeight: 600, color: color, letterSpacing: ".08em" };
    const descStyle = { fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#8b949e", textAlign: "center", marginTop: 2 };
    return (
      <div style={cell}>
        <svg width="76" height="76" viewBox="0 0 76 76">
          <circle cx="38" cy="38" r={R} fill="none" stroke="#1c2128" strokeWidth="7"
            strokeDasharray={dash + " " + C} strokeLinecap="round" transform="rotate(135 38 38)" />
          <circle cx="38" cy="38" r={R} fill="none" stroke={color} strokeWidth="7"
            strokeDasharray={dash + " " + C} strokeDashoffset={off} strokeLinecap="round"
            transform="rotate(135 38 38)" style={arc} />
          <text x="38" y="40" textAnchor="middle" fontSize="15" fontWeight="700" fill="#e6edf3"
            fontFamily="'JetBrains Mono', monospace">{isNaN(v) ? "—" : v.toFixed(2)}</text>
          <text x="38" y="54" textAnchor="middle" fontSize="8" fill="#8b949e"
            fontFamily="'JetBrains Mono', monospace">{cls.toUpperCase()}</text>
        </svg>
        <div style={nameStyle}>{spec.name}</div>
        <div style={descStyle}>{spec.full}</div>
      </div>
    );
  };

  const keys = Object.keys(SPECS).filter((k) => metrics[k] !== undefined && metrics[k] !== null);
  if (!keys.length) return null;

  const wrap = { display: "flex", flexWrap: "wrap", gap: 8, padding: "10px 12px",
    background: "#0f1117", border: "1px solid #30363d", borderRadius: 8,
    boxShadow: "0 0 0 1px rgba(29,158,117,.08), 0 8px 24px rgba(0,0,0,.4)" };
  const title = { width: "100%", fontFamily: "'JetBrains Mono', monospace", fontSize: 9,
    letterSpacing: ".18em", color: "#1D9E75", textTransform: "uppercase", marginBottom: 2 };

  return (
    <div style={wrap}>
      <div style={title}>◢ Metric Readout</div>
      {keys.map((k) => <Gauge key={k} spec={SPECS[k]} value={metrics[k]} />)}
    </div>
  );
}