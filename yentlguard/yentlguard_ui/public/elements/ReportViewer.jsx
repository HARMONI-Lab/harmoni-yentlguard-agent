import { useState } from "react";

export default function ReportViewer() {
  const p = props || {};
  const [zoom, setZoom] = useState(1);
  const [full, setFull] = useState(false);

  const html = p.html || "";
  const makeBlobUrl = () => URL.createObjectURL(new Blob([html], { type: "text/html" }));
  const openReport = () => window.open(html ? makeBlobUrl() : p.src, "_blank");
  const downloadReport = () => {
    const a = document.createElement("a");
    a.href = html ? makeBlobUrl() : p.src;
    a.download = (p.title || "report") + ".html";
    a.click();
  };

  const btnStyle = { background: "#161b22", border: "1px solid #30363d", color: "#e6edf3", borderRadius: 4, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, padding: "4px 8px", lineHeight: 1 };
  const Btn = ({ onClick, title, children }) => (
    <button onClick={onClick} title={title} style={btnStyle}>{children}</button>
  );

  const frameH = full ? "86vh" : "76vh";
  const root = { fontFamily: "'JetBrains Mono', monospace" };
  const bar = { display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 8 };
  const titleBox = { minWidth: 0 };
  const titleText = { fontSize: 12, fontWeight: 700, color: "#e6edf3", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" };
  const tsText = { fontSize: 9, color: "#8b949e" };
  const tools = { display: "flex", gap: 4, flexShrink: 0 };
  const frame = { border: "1px solid #30363d", borderRadius: 6, overflow: "hidden", background: "#fff", height: frameH };
  // Use the CSS `zoom` property (NOT transform: scale) so the iframe's layout
  // box scales with it — transform-scale leaves the hit area unscaled, which
  // makes clicks/scroll land on the wrong spot and the report feel "broken".
  const iframeStyle = { border: "none", width: "100%", height: "100%", zoom: zoom };
  // Intercept link clicks inside the report iframe from the parent (works even
  // when the report HTML has no <head> to inject into). External/root links
  // open in a new tab; in-page #anchor links keep scrolling in place. This is
  // what stops the report reloading the whole app into the iframe.
  const guardLinks = (e) => {
    try {
      const doc = e.target && e.target.contentDocument;
      if (!doc || doc.__ygLinkGuard) return;
      doc.__ygLinkGuard = true;

      // Override window.open INSIDE the iframe so any JS call (chart tooltips,
      // buttons, scripts) that would open the host app is blocked at the source.
      // External URLs are re-routed through the parent window so they still open.
      const iframeWin = doc.defaultView;
      const parentOpen = window.open.bind(window);
      iframeWin.open = (url, _name, _feat) => {
        try {
          const u = new URL(url || "", window.location.href);
          if (u.origin === window.location.origin) return null; // block host-app
          return parentOpen(u.href, "_blank", "noopener,noreferrer");
        } catch (_) { return null; }
      };

      // Also block <a> clicks that resolve to the host app
      doc.addEventListener("click", (ev) => {
        const a = ev.target && ev.target.closest ? ev.target.closest("a") : null;
        if (!a) return;
        const href = a.getAttribute("href") || "";
        if (!href || href.charAt(0) === "#") return; // let anchors scroll
        ev.preventDefault();
        try {
          const url = new URL(a.href);
          if (url.origin !== window.location.origin) {
            window.open(url.href, "_blank", "noopener,noreferrer");
          }
          // same-origin links silently blocked
        } catch (_) {}
      }, true);
    } catch (_) {}
  };

  return (
    <div style={root}>
      <div style={bar}>
        <div style={titleBox}>
          <div style={titleText}>{p.title || "Analysis report"}</div>
          <div style={tsText}>{p.timestamp || ""}</div>
        </div>
        <div style={tools}>
          <Btn title="Zoom out" onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.1).toFixed(2)))}>−</Btn>
          <Btn title="Reset zoom" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</Btn>
          <Btn title="Zoom in" onClick={() => setZoom((z) => Math.min(2, +(z + 0.1).toFixed(2)))}>+</Btn>
          <Btn title="Fullscreen" onClick={() => setFull((f) => !f)}>{full ? "▭" : "⛶"}</Btn>
          <Btn title="Open in new tab" onClick={openReport}>↗</Btn>
          <Btn title="Download" onClick={downloadReport}>⤓</Btn>
        </div>
      </div>
      <div style={frame}>
        {/* onLoad guard (above) redirects external links to a new tab while
            letting in-page #anchors scroll, so the report never reloads the
            app into this iframe. */}
        {html
          ? <iframe srcDoc={html} title="report" onLoad={guardLinks} sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms" style={iframeStyle} />
          : <iframe src={p.src} title="report" onLoad={guardLinks} sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms" style={iframeStyle} />}
      </div>
    </div>
  );
}