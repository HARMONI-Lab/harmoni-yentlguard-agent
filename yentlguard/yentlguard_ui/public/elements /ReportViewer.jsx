// public/elements/ReportViewer.jsx
// Renders the YentlGuard analysis HTML report in the ElementSidebar.
// Props: src (URL), title (string), timestamp (string)
// Props are globally injected by Chainlit — no function argument.

export default function ReportViewer() {
  const { src = "", title = "Analysis Report", timestamp = "" } = props || {};

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100%",
      width: "100%",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      background: "#0f1117",
      borderLeft: "1px solid #30363d",
    }}>
      {/* Header bar */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "10px 16px",
        borderBottom: "1px solid #30363d",
        background: "#161b22",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{
            width: "8px", height: "8px", borderRadius: "50%",
            background: "#1D9E75",
            boxShadow: "0 0 6px #1D9E7588",
            animation: "pulse 2s ease-in-out infinite",
          }} />
          <span style={{
            fontSize: "10px",
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "#1D9E75",
            fontWeight: 500,
          }}>
            {title}
          </span>
        </div>
        <span style={{
          fontSize: "10px",
          color: "#8b949e",
          letterSpacing: "0.06em",
        }}>
          {timestamp}
        </span>
      </div>

      {/* Report iframe */}
      <iframe
        src={src}
        style={{
          flex: 1,
          width: "100%",
          border: "none",
          background: "#0f1117",
        }}
        title="YentlGuard Analysis Report"
        sandbox="allow-same-origin allow-scripts"
      />

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
