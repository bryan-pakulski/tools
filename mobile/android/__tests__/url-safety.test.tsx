import { isSafeExternalUrl, validateBaseUrl } from '../src/api/urlSafety';

describe('isSafeExternalUrl', () => {
  it('allows http and https', () => {
    expect(isSafeExternalUrl('http://example.com')).toBe(true);
    expect(isSafeExternalUrl('https://example.com/path?x=1')).toBe(true);
  });

  it('blocks dangerous schemes', () => {
    for (const url of [
      'tel:+15551234567',
      'sms:+15551234567',
      'intent://example#Intent;scheme=http;end',
      'file:///etc/passwd',
      'content://media/external',
      'javascript:alert(1)',
      'data:text/html,<script>',
      'ftp://example.com',
    ]) {
      expect(isSafeExternalUrl(url)).toBe(false);
    }
  });

  it('rejects malformed and empty input', () => {
    expect(isSafeExternalUrl(null)).toBe(false);
    expect(isSafeExternalUrl(undefined)).toBe(false);
    expect(isSafeExternalUrl('')).toBe(false);
    expect(isSafeExternalUrl('not a url')).toBe(false);
  });
});

describe('validateBaseUrl', () => {
  it('accepts plain http/https and strips trailing slashes', () => {
    expect(validateBaseUrl('http://192.168.1.5:30311')).toEqual({ ok: true, url: 'http://192.168.1.5:30311' });
    expect(validateBaseUrl('https://host.example///')).toEqual({ ok: true, url: 'https://host.example' });
  });

  it('rejects embedded credentials', () => {
    const res = validateBaseUrl('https://user:token@host.example');
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error).toMatch(/username|password/i);
  });

  it('rejects non-http schemes', () => {
    expect(validateBaseUrl('ftp://host').ok).toBe(false);
    expect(validateBaseUrl('file:///etc').ok).toBe(false);
  });

  it('rejects garbage', () => {
    expect(validateBaseUrl('').ok).toBe(false);
    expect(validateBaseUrl('   ').ok).toBe(false);
    expect(validateBaseUrl('not a url').ok).toBe(false);
  });
});