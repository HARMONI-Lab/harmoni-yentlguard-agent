// YentlGuard · small, safe, additive front-end touches.
// Guarded so a missing selector never throws.
(function () {
  try {
    document.title = "YentlGuard";
  } catch (e) { /* no-op */ }
})();