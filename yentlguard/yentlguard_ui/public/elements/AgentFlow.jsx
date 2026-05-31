export default function AgentFlow() {
  const p = props || {};
  const events = p.events || [];
  const running = p.running;
  const BADGE = { BQ: "#1D9E75", RUN: "#D85A30", MCP: "#7F77DD", PHX: "#7F77DD", TOOL: "#8b949e" };

  const Spinner = ({ color }) => (
    <svg width="12" height="12" viewBox="0 0 24 24" style={ {animation: "ygspin 1s linear infinite"} }>
      <circle cx="12" cy="12" r="9" fill="none" stroke={color} strokeWidth="3"
        strokeDasharray="42 14" strokeLinecap="round" />
    </svg>
  );
  const Check = ({ color }) => (
    <svg width="12" height="12" viewBox="0 0 24 24">
      <path d="M5 13l4 4L19 7" fill="none" stroke={color} strokeWidth="3"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );

  return (
    <div style={ {fontFamily: "'JetBrains Mono', monospace", background: "#0f1117", border: "1px solid #30363d", borderRadius: 8, padding: "12px 14px", boxShadow: "0 8px 24px rgba(0,0,0,.4)"} }>
      <style>{`@keyframes ygspin{to{transform:rotate(360deg)}}
        @keyframes ygpulse{0%,100%{opacity:.4}50%{opacity:1}}`}</style>

      <div style={ {display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10} }>
        <span style={ {fontSize: 9, letterSpacing: ".18em", color: "#1D9E75", textTransform: "uppercase"} }>
          ◢ Agent Flow{" "}
          {running && <span style={ {color: "#E0A33E", animation: "ygpulse 1.2s infinite"} }>● live</span>}
        </span>
        <span style={ {fontSize: 9, color: "#8b949e"} }>
          {(p.agents || 0)} agents · {(p.tools || 0)} tools · {(p.elapsed || 0)}s
        </span>
      </div>

      <div style={ {position: "relative", paddingLeft: 18} }>
        <div style={ {position: "absolute", left: 5, top: 4, bottom: 4, width: 2, background: "#1c2128"} } />
        {events.length === 0 && (
          <div style={ {fontSize: 11, color: "#8b949e"} }>Waiting for first event…</div>
        )}
        {events.map((e, i) => {
          const isAgent = e.kind === "agent";
          const color = isAgent ? "#e6edf3" : (BADGE[e.badge] || BADGE.TOOL);
          return (
            <div key={i} style={ {position: "relative", marginBottom: 10} }>
              <div style={{ position: "absolute", left: -16, top: 1, width: 12, height: 12,
                borderRadius: "50%", background: "#0f1117", border: `2px solid ${color}`,
                boxShadow: e.status === "running" ? `0 0 8px ${color}` : "none" }} />
              {isAgent ? (
                <div style={ {fontSize: 11, fontWeight: 700, color: "#e6edf3", letterSpacing: ".04em"} }>
                  ↳ {e.agent}
                </div>
              ) : (
                <div style={ {display: "flex", alignItems: "center", gap: 8} }>
                  <span style={ {fontSize: 8, fontWeight: 700, color: "#000", background: color, padding: "1px 5px", borderRadius: 3, letterSpacing: ".06em"} }>{e.badge}</span>
                  <span style={ {fontSize: 11, color: "#e6edf3"} }>{e.label}</span>
                  <span style={ {marginLeft: "auto", display: "flex", alignItems: "center", gap: 6} }>
                    {e.duration != null && <span style={ {fontSize: 9, color: "#8b949e"} }>{e.duration}s</span>}
                    {e.status === "running" ? <Spinner color={color} />
                      : e.status === "error" ? <span style={ {color: "#D85A30", fontSize: 11} }>✕</span>
                      : <Check color={color} />}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}