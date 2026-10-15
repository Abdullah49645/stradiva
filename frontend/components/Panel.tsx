export default function Panel({
  children,
  className = "",
  reticleColor,
  as: Tag = "div",
}: {
  children: React.ReactNode;
  className?: string;
  reticleColor?: string;
  as?: "div" | "section";
}) {
  const style = reticleColor ? ({ "--reticle-color": reticleColor } as React.CSSProperties) : undefined;
  return (
    <Tag
      className={`reticle glass-panel rounded-md border border-hairline bg-graphite p-6 ${className}`}
      style={style}
    >
      {children}
    </Tag>
  );
}
