/** @type {import('next').NextConfig} */
// Static export: the UI is plain files (out/) that S3 + CloudFront, Amplify Hosting or any static host can serve.
// The API base is baked in at build time from NEXT_PUBLIC_STRADIVA_API_BASE (empty = bundled case files only).
const nextConfig = { output: "export", images: { unoptimized: true } };
module.exports = nextConfig;
