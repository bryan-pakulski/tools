import React from 'react';
import { Linking, Alert, Text } from 'react-native';

/**
 * Safe external URL opening. Server/agent-controlled URLs (research sources,
 * markdown links, artifact navigation) must never reach Linking.openURL
 * unfiltered: tel:/sms:/intent:/file:/content: and app deep-link schemes can
 * trigger actions outside the app. Only http/https pass.
 */
export function isSafeExternalUrl(url: string | null | undefined): boolean {
  if (!url) return false;
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

/**
 * Open a URL only when it is http(s). Unknown/unsafe schemes are surfaced to
 * the user (once per attempt) instead of being handed to the OS handler.
 * Returns true when the URL was actually opened.
 */
export async function openExternalUrl(url: string | null | undefined): Promise<boolean> {
  if (!isSafeExternalUrl(url)) {
    if (url) {
      Alert.alert('Link blocked', 'Only http:// and https:// links can be opened.');
    }
    return false;
  }
  try {
    await Linking.openURL(url as string);
    return true;
  } catch {
    Alert.alert('Could not open link', url as string);
    return false;
  }
}

/**
 * Validate a backend base URL from user input. Rejects embedded credentials
 * (user:pass@host), non-http(s) schemes, and whitespace — those were
 * previously accepted by a loose prefix check and stored in plaintext.
 * Returns the normalized URL (no trailing slash) or an error string.
 */
export function validateBaseUrl(raw: string): { ok: true; url: string } | { ok: false; error: string } {
  const trimmed = raw.trim();
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return { ok: false, error: 'URL must use http:// or https://.' };
    }
    if (parsed.username || parsed.password) {
      return { ok: false, error: 'URLs with embedded usernames or passwords are not allowed.' };
    }
    return { ok: true, url: trimmed.replace(/\/+$/, '') };
  } catch {
    return { ok: false, error: 'Enter a valid http:// or https:// URL.' };
  }
}

/**
 * Markdown link rule for react-native-markdown-display (RenderLinkFunction
 * signature): routes link taps through openExternalUrl so tel:/intent:/
 * custom schemes in agent-produced markdown never reach the OS handler.
 */
export function makeSafeLinkRule(
  node: { key?: string; attributes?: { href?: string } },
  children: React.ReactNode[],
  _parentNodes: unknown[],
  styles?: unknown,
): React.ReactNode {
  return (
    <Text key={node.key} style={[styles as { [k: string]: unknown } | undefined, { textDecorationLine: 'underline' }]}>
      {children}
    </Text>
  );
}
