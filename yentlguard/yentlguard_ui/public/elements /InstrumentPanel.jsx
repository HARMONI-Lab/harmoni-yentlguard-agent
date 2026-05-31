// public/elements/InstrumentPanel.jsx
// Shown in the ElementSidebar before any analysis report is generated.
// Displays system status and configuration as a scientific instrument readout.
// Props: status, model, project, phoenix

export default function InstrumentPanel() {
  const {
    status = "READY",
    model = "gemini-2.5-pro",
    project = "—",
    phoenix = "—",
  } = props || {};

  const rows = [
    { label: "STATUS",  value: status,  accent: "#1D9E75" },
    { label: "MODEL",   value: model,   accent: "#e6edf3" },
    { label: "PROJECT", value: project, accent: "#e6edf3" },
    { label: "PHOENIX", value: phoenix.replace("https://", ""), accent: "#7F77DD" },
  ];

  return (
    <div style={{
      height: "100%",
      width: "100%",
      background: "#0f1117",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: "0",
      padding: "48px 32px",
      boxSizing: "border-box",
      borderLeft: "1px solid #30363d",
    }}>

      {/* Logo mark */}
      <div style={{
        marginBottom: "48px",
        textAlign: "center",
      }}>
        <div style={{
          fontSize: "11px",
          letterSpacing: "0.22em",
          color: "#1D9E75",
          textTransform: "uppercase",
          marginBottom: "8px",
        }}>
          HARMONI Lab
        </div>
        <div style={{
          fontSize: "28px",
          fontWeight: 700,
          color: "#e6edf3",
          letterSpacing: "0.04em",
          lineHeight: 1.1,
        }}>
          YENTL<span style={{ color: "#1D9E75" }}>GUARD</span>
        </div>
        <div style={{
          fontSize: "10px",
          color: "#8b949e",
          letterSpacing: "0.12em",
          marginTop: "6px",
          textTransform: "uppercase",
        }}>
          Mechanistic Interpretability · Clinical Triage
        </div>
      </div>

      {/* Instrument rows */}
      <div style={{
        width: "100%",
        maxWidth: "320px",
        display: "flex",
        flexDirection: "column",
        gap: "0",
        border: "1px solid #30363d",
        borderRadius: "4px",
        overflow: "hidden",
      }}>
        {rows.map((row, i) => (
          <div key={row.label} style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "11px 16px",
            borderBottom: i < rows.length - 1 ? "1px solid #30363d" : "none",
            background: i % 2 === 0 ? "#161b22" : "#0f1117",
          }}>
            <span style={{
              fontSize: "10px",
              letterSpacing: "0.12em",
              color: "#8b949e",
              textTransform: "uppercase",
            }}>
              {row.label}
            </span>
            <span style={{
              fontSize: "11px",
              color: row.accent,
              letterSpacing: "0.04em",
              maxWidth: "180px",
              textAlign: "right",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}>
              {row.value}
            </span>
          </div>
        ))}
      </div>

      {/* Waiting indicator */}
      <div style={{
        marginTop: "40px",
        display: "flex",
        alignItems: "center",
        gap: "8px",
      }}>
        <div style={{
          width: "6px", height: "6px", borderRadius: "50%",
          background: "#1D9E75",
          animation: "blink 1.4s step-start infinite",
        }} />
        <div style={{
          width: "6px", height: "6px", borderRadius: "50%",
          background: "#1D9E75",
          animation: "blink 1.4s step-start 0.2s infinite",
        }} />
        <div style={{
          width: "6px", height: "6px", borderRadius: "50%",
          background: "#1D9E75",
          animation: "blink 1.4s step-start 0.4s infinite",
        }} />
        <span style={{
          fontSize: "10px",
          color: "#8b949e",
          letterSpacing: "0.1em",
          marginLeft: "4px",
          textTransform: "uppercase",
        }}>
          Awaiting run
        </span>
      </div>

      <div style={{
        marginTop: "48px",
        fontSize: "10px",
        color: "#30363d",
        letterSpacing: "0.08em",
        textAlign: "center",
        lineHeight: 1.7,
      }}>
        Run analyze_run to load a report<br />
        into this panel
      </div>

      <style>{`
        @keyframes blink {
          0%, 100% { opacity: 0.2; }
          20% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
