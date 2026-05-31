// public/elements/MetricPulse.jsx
// Inline metric strip shown below agent messages that contain ΔM / CRR / TAR values.
// Props: metrics — { delta_m?, crr?, tar?, gap?, pss? }

export default function MetricPulse() {
  const { metrics = {} } = props || {};

  const definitions = [
    {
      key: "delta_m",
      label: "ΔM",
      desc: "Token Confidence Margin",
      color: "#1D9E75",
      threshold: { warn: 1.0, dir: "below" },
    },
    {
      key: "crr",
      label: "CRR",
      desc: "Confidence Recovery Rate",
      color: "#7F77DD",
      threshold: { warn: 0.1, ok: 0.95, dir: "range" },
    },
    {
      key: "tar",
      label: "TAR",
      desc: "Thought Allocation Ratio",
      color: "#e6edf3",
      threshold: { warn: 2.0, dir: "above" },
    },
    {
      key: "gap",
      label: "GAP",
      desc: "Sycophancy Gap",
      color: "#D85A30",
      threshold: { warn: 0.1, ok: 0.3, dir: "range" },
    },
    {
      key: "pss",
      label: "PSS",
      desc: "Perturbation Sensitivity",
      color: "#e6edf3",
      threshold: null,
    },
  ];

  const active = definitions.filter(d => metrics[d.key] !== undefined);
  if (active.length === 0) return null;

  function statusColor(def) {
    const v = parseFloat(metrics[def.key]);
    if (!def.threshold || isNaN(v)) return def.color;
    const { warn, ok, dir } = def.threshold;
    if (dir === "below") return v < warn ? "#D85A30" : def.color;
    if (dir === "above") return v > warn ? "#D85A30" : def.color;
    if (dir === "range") {
      if (ok && v >= ok) return "#1D9E75";
      if (v < warn) return "#D85A30";
      return "#f0a94a";
    }
    return def.color;
  }

  function verdict(def) {
    const v = parseFloat(metrics[def.key]);
    if (isNaN(v)) return null;
    const { warn, ok, dir } = def.threshold || {};
    if (def.key === "crr") {
      if (v >= 0.95) return "full recovery";
      if (v >= 0.1) return "partial";
      return "failed";
    }
    if (def.key === "gap") {
      if (v > 0.3) return "genuine debiasing";
      if (v < 0.1) return "likely sycophancy";
      return "ambiguous";
    }
    if (def.key === "delta_m" && v < 1.0) return "low confidence";
    if (def.key === "tar" && v > 2.0) return "high friction";
    return null;
  }

  return (
    <div style={{
      display: "flex",
      gap: "8px",
      flexWrap: "wrap",
      margin: "4px 0 2px",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    }}>
      {active.map(def => {
        const sc = statusColor(def);
        const v = verdict(def);
        return (
          <div key={def.key} title={def.desc} style={{
            display: "flex",
            alignItems: "baseline",
            gap: "5px",
            padding: "5px 10px",
            background: "#161b22",
            border: `1px solid ${sc}44`,
            borderLeft: `3px solid ${sc}`,
            borderRadius: "3px",
            cursor: "default",
          }}>
            <span style={{
              fontSize: "9px",
              letterSpacing: "0.12em",
              color: "#8b949e",
              textTransform: "uppercase",
            }}>
              {def.label}
            </span>
            <span style={{
              fontSize: "14px",
              color: sc,
              fontWeight: 500,
              letterSpacing: "0.02em",
            }}>
              {parseFloat(metrics[def.key]).toFixed(3)}
            </span>
            {v && (
              <span style={{
                fontSize: "8px",
                color: sc,
                opacity: 0.7,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}>
                {v}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
