import type { Metadata } from "next";
import "@fontsource/space-grotesk/500.css";
import "@fontsource/space-grotesk/600.css";
import "@fontsource/space-grotesk/700.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "CARAI — Reconstruction & Avoidability Analysis",
  description: "Computer-vision accident reconstruction and physical avoidability analysis. OpenCV AI Competition 2026.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-asphalt text-paper min-h-screen font-body antialiased">
        <div className="hud-backdrop" aria-hidden="true" />
        {children}
      </body>
    </html>
  );
}
