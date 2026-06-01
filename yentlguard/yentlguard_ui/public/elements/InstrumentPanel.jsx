export default function InstrumentPanel() {
  const p = props || {};
  const rows = [
    ["MODEL", p.model || "—"],
    ["PROJECT", p.project || "—"],
    ["PHOENIX", p.phoenix || "—"],
    ["STATUS", p.status || "READY"],
  ];
  const LEGEND = [
    ["BQ", "#1D9E75", "BigQuery"],
    ["RUN", "#D85A30", "Runner"],
    ["MCP", "#7F77DD", "Phoenix MCP"],
    ["PHX", "#7F77DD", "Phoenix fn"],
  ];

  const root = { fontFamily: "'JetBrains Mono', monospace", color: "#e6edf3", padding: "8px 4px" };
  const brandRow = { display: "flex", alignItems: "center", gap: 10, marginBottom: 6 };
  const dot = { width: 10, height: 10, borderRadius: "50%", background: "#1D9E75", boxShadow: "0 0 10px #1D9E75", animation: "ygpulse 1.6s infinite" };
  const brand = { fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 20, letterSpacing: ".04em" };
  const sub = { fontSize: 10, color: "#8b949e", marginBottom: 18, letterSpacing: ".06em" };
  const table = { border: "1px solid #30363d", borderRadius: 6, overflow: "hidden" };
  const legendHead = { marginTop: 18, fontSize: 9, letterSpacing: ".16em", color: "#8b949e", textTransform: "uppercase", marginBottom: 8 };
  const legendWrap = { display: "flex", flexWrap: "wrap", gap: 6 };
  const footer = { marginTop: 26, display: "flex", alignItems: "center", gap: 8, fontSize: 10, color: "#8b949e" };
  const footDot = { width: 6, height: 6, borderRadius: "50%", background: "#E0A33E", animation: "ygpulse 1.2s infinite" };

  return (
    <div style={root}>
      <style>{`@keyframes ygpulse{0%,100%{opacity:.35}50%{opacity:1}}`}</style>
      <div style={brandRow}>
        <div style={dot} />
        <div style={brand}>YENTL<span style={ {color: "#1D9E75"} }>GUARD</span></div>
      </div>
      <div style={sub}>Mechanistic Interpretability · Clinical Triage</div>

      <div style={table}>
        {rows.map((r, i) => {
          const row = { display: "flex", justifyContent: "space-between", padding: "9px 12px", fontSize: 11, background: i % 2 ? "#0f1117" : "#161b22", borderBottom: i < rows.length - 1 ? "1px solid #30363d" : "none" };
          const kStyle = { color: "#8b949e", letterSpacing: ".12em" };
          const vStyle = { color: r[0] === "STATUS" ? "#1D9E75" : "#e6edf3", fontWeight: 600, maxWidth: 170, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" };
          return (
            <div key={r[0]} style={row}>
              <span style={kStyle}>{r[0]}</span>
              <span style={vStyle}>{r[1]}</span>
            </div>
          );
        })}
      </div>

      <div style={legendHead}>Tool families</div>
      <div style={legendWrap}>
        {LEGEND.map((l) => {
          const item = { display: "inline-flex", alignItems: "center", gap: 5, fontSize: 9, color: "#8b949e" };
          const chip = { background: l[1], color: "#000", fontWeight: 700, padding: "1px 5px", borderRadius: 3 };
          return (
            <span key={l[0]} style={item}>
              <span style={chip}>{l[0]}</span>
              {l[2]}
            </span>
          );
        })}
      </div>

      <div style={footer}>
        <div style={footDot} />
        Awaiting first analysis — run <span style={ {color: "#D85A30"} }>analyze_run</span> to populate.
      </div>
    </div>
  );
}