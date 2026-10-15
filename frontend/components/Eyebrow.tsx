export default function Eyebrow({ children, color = "text-hud" }: { children: React.ReactNode; color?: string }) {
  return (
    <div className={`font-mono text-[11px] uppercase tracking-[0.14em] ${color} flex items-center gap-2 whitespace-nowrap`}>
      <span className="inline-block w-3 h-px bg-current shrink-0" />
      {children}
    </div>
  );
}
