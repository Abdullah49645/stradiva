import type { Metadata } from "next";
import "@fontsource/overpass/500.css";
import "@fontsource/overpass/700.css";
import "@fontsource/overpass/800.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "CARAI — evidence from traffic video",
  description:
    "Recovers vehicle motion from uncalibrated traffic footage, cross-examines its own reconstruction, and states what the video can and cannot establish.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
