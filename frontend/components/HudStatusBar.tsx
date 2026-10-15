export default function HudStatusBar({ status }: { status?: string | null }) {
  return (
    <div className="sticky top-0 z-20 -mx-5 sm:-mx-8 mb-10 border-b border-hairline bg-asphalt/80 backdrop-blur-md">
      <div className="max-w-5xl mx-auto px-5 sm:px-8 py-2.5 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
        <div className="flex items-center gap-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-hud pulse-dot" />
          <span className="text-paper">CARAI</span>
          <span className="text-muted/60">// reconstruction &amp; avoidability engine</span>
        </div>
        <div className="hidden sm:flex items-center gap-4 text-muted/70">
          <span>reconstructor · skeptic</span>
          <span className="w-px h-3 bg-hairline" />
          <span>{status ? status.replace(/_/g, " ") : "idle"}</span>
        </div>
      </div>
    </div>
  );
}
