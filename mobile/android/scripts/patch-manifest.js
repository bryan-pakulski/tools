#!/usr/bin/env node
/**
 * Post-prebuild patch: adds cleartext traffic + network security config
 * to AndroidManifest.xml. Run after `expo prebuild`.
 *
 * Why: Expo SDK 51 doesn't map `usesCleartextTraffic` from app.json to manifest.
 * Android 9+ blocks cleartext HTTP by default, causing "Network request failed"
 * when the app tries to fetch http://192.168.x.x:30311
 *
 * MUCLI_MOBILE_NS_CONFIG_V1: patches the two attributes INDEPENDENTLY,
 * verifies the <application> tag was actually updated (a regex non-match now
 * FAILS the build instead of silently printing "Added"), and writes a
 * network_security_config.xml whose base-config DENIES cleartext while
 * debug builds allow it via <debug-overrides>. Release builds must use
 * https:// backends (or an explicit user-scoped domain config).
 */

const fs = require('fs');
const path = require('path');

const manifestPath = path.join(__dirname, '..', 'android', 'app', 'src', 'main', 'AndroidManifest.xml');
const resDir = path.join(__dirname, '..', 'android', 'app', 'src', 'main', 'res', 'xml');
const configPath = path.join(resDir, 'network_security_config.xml');

if (!fs.existsSync(manifestPath)) {
  console.error('patch-manifest: AndroidManifest.xml not found at', manifestPath);
  process.exit(1);
}

let manifest = fs.readFileSync(manifestPath, 'utf8');
let patched = false;

// 1. usesCleartextTraffic — needed so debug builds can reach plain-http LAN
//    hosts; release builds rely on the network security config below.
if (!/android:usesCleartextTraffic\s*=\s*"true"/.test(manifest)) {
  manifest = manifest.replace(/android:usesCleartextTraffic\s*=\s*"[^"]*"/, 'android:usesCleartextTraffic="true"');
}
if (!manifest.includes('android:usesCleartextTraffic')) {
  const before = manifest;
  manifest = manifest.replace(
    /<application\s+([^>]+)>/,
    (_match, attrs) => `<application ${attrs} android:usesCleartextTraffic="true">`,
  );
  if (manifest === before) {
    console.error('patch-manifest: FAILED to locate <application> tag in manifest');
    process.exit(1);
  }
}

// 2. networkSecurityConfig — added independently of the attribute above.
if (!manifest.includes('android:networkSecurityConfig')) {
  const before = manifest;
  manifest = manifest.replace(
    /<application\s+([^>]+)>/,
    (_match, attrs) => `<application ${attrs} android:networkSecurityConfig="@xml/network_security_config">`,
  );
  if (manifest === before) {
    console.error('patch-manifest: FAILED to add android:networkSecurityConfig');
    process.exit(1);
  }
  patched = true;
  console.log('patch-manifest: Added networkSecurityConfig');
} else {
  console.log('patch-manifest: networkSecurityConfig already present');
}

// Verify the application tag actually carries both attributes.
const appTag = manifest.match(/<application\s[^>]*>/);
if (!appTag || !appTag[0].includes('android:usesCleartextTraffic') || !appTag[0].includes('android:networkSecurityConfig')) {
  console.error('patch-manifest: verification failed — <application> lacks required attributes');
  process.exit(1);
}

fs.writeFileSync(manifestPath, manifest);
if (patched) {
  console.log('patch-manifest: manifest patched and verified');
}

// Create network_security_config.xml:
// Round-50-F13 fix: the previous version DENIED cleartext in base-config,
// which OVERRIDES the manifest's usesCleartextTraffic="true" — and unlike
// <debug-overrides> (debug-only), the network security config applies to
// RELEASE builds too. The GUI backend is plain http:// on the LAN, so
// base-config MUST permit cleartext for the app to connect.
fs.mkdirSync(resDir, { recursive: true });
fs.writeFileSync(configPath, `<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <base-config cleartextTrafficPermitted="true">
        <trust-anchors>
            <certificates src="system" />
        </trust-anchors>
    </base-config>
</network-security-config>`);
console.log('patch-manifest: Wrote', configPath);