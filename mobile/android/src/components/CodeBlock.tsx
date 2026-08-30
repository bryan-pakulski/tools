import React, { useCallback, useMemo, useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Clipboard from 'expo-clipboard';
import type { SyntaxColors } from '../theme/tokens';
import { spacing } from '../theme/tokens';

// ── Token types ───────────────────────────────────────────────────────────

type TokenType = 'keyword' | 'string' | 'comment' | 'number' | 'func' | 'operator' | 'punctuation' | 'plain' | 'added' | 'removed' | 'diffHeader';

interface Token {
  text: string;
  type: TokenType;
}

// ── Language detection ────────────────────────────────────────────────────

const LANG_ALIASES: Record<string, string> = {
  py: 'python', python: 'python',
  js: 'javascript', javascript: 'javascript', jsx: 'javascript',
  ts: 'typescript', typescript: 'typescript', tsx: 'typescript',
  sh: 'bash', bash: 'bash', shell: 'bash', zsh: 'bash',
  json: 'json',
  yaml: 'yaml', yml: 'yaml',
  html: 'html', xml: 'html',
  css: 'css',
  sql: 'sql',
  diff: 'diff', patch: 'diff',
  go: 'go',
  rust: 'rust', rs: 'rust',
  java: 'java',
  cpp: 'cpp', 'c++': 'cpp', c: 'cpp',
  ruby: 'ruby', rb: 'ruby',
  php: 'php',
  md: 'markdown', markdown: 'markdown',
};

function normalizeLang(lang: string): string {
  return LANG_ALIASES[lang.toLowerCase().trim()] || 'plain';
}

// ── Keyword sets ──────────────────────────────────────────────────────────

const KEYWORDS: Record<string, Set<string>> = {
  python: new Set(['def', 'class', 'import', 'from', 'as', 'if', 'elif', 'else', 'try', 'except', 'finally', 'with', 'for', 'while', 'return', 'yield', 'raise', 'pass', 'break', 'continue', 'lambda', 'global', 'nonlocal', 'assert', 'del', 'in', 'is', 'not', 'and', 'or', 'True', 'False', 'None', 'self', 'async', 'await', 'print']),
  javascript: new Set(['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'new', 'delete', 'typeof', 'instanceof', 'in', 'of', 'this', 'super', 'class', 'extends', 'static', 'get', 'set', 'async', 'await', 'yield', 'import', 'from', 'export', 'default', 'try', 'catch', 'finally', 'throw', 'void', 'null', 'undefined', 'true', 'false', 'typeof']),
  typescript: new Set(['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'new', 'delete', 'typeof', 'instanceof', 'in', 'of', 'this', 'super', 'class', 'extends', 'static', 'get', 'set', 'async', 'await', 'yield', 'import', 'from', 'export', 'default', 'try', 'catch', 'finally', 'throw', 'void', 'null', 'undefined', 'true', 'false', 'typeof', 'type', 'interface', 'enum', 'namespace', 'declare', 'readonly', 'public', 'private', 'protected', 'abstract', 'implements', 'as', 'is', 'keyof', 'infer', 'never', 'unknown', 'any']),
  bash: new Set(['if', 'then', 'else', 'elif', 'fi', 'for', 'do', 'done', 'while', 'case', 'esac', 'in', 'function', 'return', 'exit', 'break', 'continue', 'local', 'export', 'unset', 'source', 'echo', 'printf', 'read', 'set', 'shift', 'trap', 'test', 'true', 'false']),
  go: new Set(['func', 'var', 'const', 'type', 'struct', 'interface', 'package', 'import', 'return', 'if', 'else', 'for', 'range', 'switch', 'case', 'default', 'break', 'continue', 'go', 'defer', 'chan', 'select', 'map', 'make', 'new', 'nil', 'true', 'false']),
  rust: new Set(['fn', 'let', 'mut', 'const', 'static', 'struct', 'enum', 'trait', 'impl', 'pub', 'use', 'mod', 'crate', 'self', 'super', 'as', 'in', 'ref', 'match', 'if', 'else', 'for', 'while', 'loop', 'return', 'break', 'continue', 'async', 'await', 'move', 'dyn', 'unsafe', 'where', 'type', 'true', 'false', 'None', 'Some', 'Ok', 'Err']),
  java: new Set(['public', 'private', 'protected', 'static', 'final', 'class', 'interface', 'extends', 'implements', 'import', 'package', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'new', 'this', 'super', 'try', 'catch', 'finally', 'throw', 'throws', 'void', 'int', 'long', 'double', 'float', 'boolean', 'char', 'byte', 'short', 'true', 'false', 'null', 'instanceof', 'synchronized', 'abstract', 'enum']),
  cpp: new Set(['int', 'long', 'double', 'float', 'char', 'bool', 'void', 'const', 'static', 'class', 'struct', 'public', 'private', 'protected', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'new', 'delete', 'this', 'try', 'catch', 'throw', 'namespace', 'using', 'template', 'typename', 'true', 'false', 'nullptr', 'include', 'define', 'ifdef', 'ifndef', 'endif']),
  ruby: new Set(['def', 'end', 'class', 'module', 'if', 'elsif', 'else', 'unless', 'while', 'until', 'for', 'do', 'break', 'next', 'redo', 'retry', 'return', 'yield', 'begin', 'rescue', 'ensure', 'raise', 'throw', 'catch', 'require', 'require_relative', 'load', 'include', 'extend', 'attr_accessor', 'attr_reader', 'attr_writer', 'self', 'super', 'nil', 'true', 'false', 'puts', 'print', 'lambda', 'proc']),
  php: new Set(['function', 'class', 'interface', 'extends', 'implements', 'public', 'private', 'protected', 'static', 'const', 'return', 'if', 'else', 'elseif', 'for', 'foreach', 'while', 'do', 'switch', 'case', 'break', 'continue', 'new', 'this', 'self', 'parent', 'try', 'catch', 'finally', 'throw', 'use', 'namespace', 'require', 'require_once', 'include', 'include_once', 'echo', 'print', 'true', 'false', 'null', 'array', 'isset', 'unset']),
  sql: new Set(['SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'CREATE', 'TABLE', 'ALTER', 'DROP', 'INDEX', 'VIEW', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'AS', 'AND', 'OR', 'NOT', 'NULL', 'IS', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'ORDER', 'BY', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'DISTINCT', 'UNION', 'ALL', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'DEFAULT', 'UNIQUE', 'CONSTRAINT']),
};

function getKeywords(lang: string): Set<string> {
  return KEYWORDS[lang] || new Set();
}

// ── Tokenizers ────────────────────────────────────────────────────────────

/**
 * Generic tokenizer that handles common syntax patterns:
 * comments, strings, numbers, function calls, keywords, operators.
 *
 * Each language can override comment patterns. The algorithm is a
 * single-pass scanner that tries patterns at each position.
 */

function tokenizeLine(line: string, lang: string, keywords: Set<string>): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const len = line.length;

  // Comment prefixes by language
  const lineCommentPrefixes: Record<string, string[]> = {
    python: ['#'],
    javascript: ['//'],
    typescript: ['//'],
    bash: ['#'],
    go: ['//'],
    rust: ['//'],
    java: ['//'],
    cpp: ['//'],
    ruby: ['#'],
    php: ['//', '#'],
    sql: ['--'],
    css: ['/*'],  // handled specially
    html: [],     // no line comments
    json: [],
    yaml: ['#'],
    markdown: [],
  };

  while (i < len) {
    // ── Line comments ───────────────────────────────────────────
    const prefixes = lineCommentPrefixes[lang] || [];
    let matchedComment = false;
    for (const prefix of prefixes) {
      if (line.substr(i, prefix.length) === prefix) {
        tokens.push({ text: line.slice(i), type: 'comment' });
        return tokens;
      }
    }

    // ── Strings (single, double, backtick, triple) ──────────────
    const ch = line[i];
    if (ch === '"' || ch === "'" || ch === '`') {
      // Triple-quoted (python)
      if (lang === 'python' && line.substr(i, 3) === '"""' || line.substr(i, 3) === "'''") {
        const quote = line.substr(i, 3);
        const end = line.indexOf(quote, i + 3);
        const stop = end === -1 ? len : end + 3;
        tokens.push({ text: line.slice(i, stop), type: 'string' });
        i = stop;
        continue;
      }
      // Regular string — scan to closing quote (allow escapes)
      let j = i + 1;
      while (j < len) {
        if (line[j] === '\\') { j += 2; continue; }
        if (line[j] === ch) { j++; break; }
        j++;
      }
      tokens.push({ text: line.slice(i, j), type: 'string' });
      i = j;
      continue;
    }

    // ── Numbers ─────────────────────────────────────────────────
    if (/[0-9]/.test(ch) || (ch === '-' && i + 1 < len && /[0-9]/.test(line[i + 1]) && (tokens.length === 0 || tokens[tokens.length - 1].type === 'operator' || tokens[tokens.length - 1].type === 'punctuation'))) {
      let j = i;
      if (ch === '-') j++;
      while (j < len && /[0-9a-fA-FxXoObB._eE+\-]/.test(line[j])) {
        // Stop if it's clearly not part of a number (e.g., followed by a letter that's not e/E in float context)
        if (j > i + 1 && /[a-df-wzA-DF-WZ]/.test(line[j]) && line[j] !== 'e' && line[j] !== 'E') break;
        j++;
      }
      tokens.push({ text: line.slice(i, j), type: 'number' });
      i = j;
      continue;
    }

    // ── Identifiers / keywords ──────────────────────────────────
    if (/[a-zA-Z_$@]/.test(ch)) {
      let j = i;
      while (j < len && /[a-zA-Z0-9_$]/.test(line[j])) j++;
      const word = line.slice(i, j);
      // Check if function call (followed by optional spaces then `(`)
      let k = j;
      while (k < len && line[k] === ' ') k++;
      const isFunc = line[k] === '(' && !keywords.has(word);
      const type: TokenType = keywords.has(word) ? 'keyword' : (isFunc ? 'func' : 'plain');
      tokens.push({ text: word, type });
      i = j;
      continue;
    }

    // ── Operators ───────────────────────────────────────────────
    if (/[+\-*/%=<>!&|^~?:]/.test(ch)) {
      let j = i;
      while (j < len && /[+\-*/%=<>!&|^~?:]/.test(line[j])) j++;
      tokens.push({ text: line.slice(i, j), type: 'operator' });
      i = j;
      continue;
    }

    // ── Punctuation ─────────────────────────────────────────────
    if (/[(){}\[\];,.]/.test(ch)) {
      tokens.push({ text: ch, type: 'punctuation' });
      i++;
      continue;
    }

    // ── Whitespace + everything else ────────────────────────────
    let j = i;
    while (j < len && /\s/.test(line[j])) j++;
    if (j > i) {
      tokens.push({ text: line.slice(i, j), type: 'plain' });
      i = j;
      continue;
    }
    // Single char fallback
    tokens.push({ text: ch, type: 'plain' });
    i++;
  }

  return tokens;
}

function tokenizeDiff(line: string): Token {
  if (line.startsWith('@@')) return { text: line, type: 'diffHeader' };
  if (line.startsWith('+++') || line.startsWith('---')) return { text: line, type: 'diffHeader' };
  if (line.startsWith('+')) return { text: line, type: 'added' };
  if (line.startsWith('-')) return { text: line, type: 'removed' };
  return { text: line, type: 'plain' };
}

function tokenizeYamlLine(line: string, keywords: Set<string>): Token[] {
  // YAML key: value — highlight the key
  const keyMatch = line.match(/^(\s*)([\w.-]+)(:)(\s*)(.*)$/);
  if (keyMatch) {
    const [, indent, key, colon, space, rest] = keyMatch;
    const tokens: Token[] = [
      { text: indent, type: 'plain' },
      { text: key, type: 'keyword' },
      { text: colon, type: 'punctuation' },
      { text: space, type: 'plain' },
    ];
    // Tokenize the value part
    if (rest.startsWith('#')) {
      tokens.push({ text: rest, type: 'comment' });
    } else if (rest.startsWith('"') || rest.startsWith("'")) {
      tokens.push({ text: rest, type: 'string' });
    } else if (/^-?\d/.test(rest)) {
      tokens.push({ text: rest, type: 'number' });
    } else if (rest === 'true' || rest === 'false' || rest === 'null' || rest === '~') {
      tokens.push({ text: rest, type: 'number' });
    } else if (rest) {
      tokens.push({ text: rest, type: 'string' });
    }
    return tokens;
  }
  if (line.trimStart().startsWith('#')) {
    return [{ text: line, type: 'comment' }];
  }
  if (line.trimStart().startsWith('- ')) {
    const indent = line.match(/^(\s*)/)![0];
    return [
      { text: indent, type: 'plain' },
      { text: '- ', type: 'operator' },
      ...tokenizeLine(line.trimStart().slice(2), 'yaml', keywords),
    ];
  }
  return [{ text: line, type: 'plain' }];
}

// ── Main tokenizer ────────────────────────────────────────────────────────

function tokenize(code: string, lang: string): Token[][] {
  const normalized = normalizeLang(lang);
  const keywords = getKeywords(normalized);
  const lines = code.split('\n');

  return lines.map(line => {
    if (normalized === 'diff') {
      return [tokenizeDiff(line)];
    }
    if (normalized === 'yaml') {
      return tokenizeYamlLine(line, keywords);
    }
    if (normalized === 'plain') {
      return [{ text: line, type: 'plain' }];
    }
    return tokenizeLine(line, normalized, keywords);
  });
}

// ── Component ─────────────────────────────────────────────────────────────

interface CodeBlockProps {
  code: string;
  language?: string;
  colors: { bgHover: string; textSoft: string; textDim: string; syntax: SyntaxColors };
}

const LINE_HEIGHT = 19;
const VISIBLE_LINES = 25;
const OVERSCAN = 8;

export function CodeBlock({ code, language, colors }: CodeBlockProps) {
  const langLabel = language ? language.toLowerCase().trim() : '';
  const tokenizedLines = useMemo(() => tokenize(code, langLabel), [code, langLabel]);
  const totalLines = tokenizedLines.length;
  const needsWindowing = totalLines > VISIBLE_LINES + OVERSCAN * 2;
  const [scrollY, setScrollY] = useState(0);

  const copy = useCallback(() => {
    Clipboard.setStringAsync(code);
  }, [code]);

  // Visible window: render only lines within [start, end) + overscan.
  // Spacer Views above/below maintain total scroll height so the scrollbar
  // and scroll position reflect the full content without mounting thousands
  // of native <Text> nodes for large code dumps.
  let startIdx = 0;
  let endIdx = totalLines;
  let topSpacerHeight = 0;
  let bottomSpacerHeight = 0;
  if (needsWindowing) {
    const firstVisible = Math.floor(scrollY / LINE_HEIGHT);
    startIdx = Math.max(0, firstVisible - OVERSCAN);
    endIdx = Math.min(totalLines, startIdx + VISIBLE_LINES + OVERSCAN * 2);
    topSpacerHeight = startIdx * LINE_HEIGHT;
    bottomSpacerHeight = (totalLines - endIdx) * LINE_HEIGHT;
  }

  const onScroll = useCallback((e: any) => {
    if (!e?.nativeEvent?.contentOffset) return;
    if (!needsWindowing) return;
    setScrollY(e.nativeEvent.contentOffset.y);
  }, [needsWindowing]);

  return (
    <View style={[styles.container, { backgroundColor: colors.bgHover }]}>
      <View style={styles.header}>
        {langLabel ? (
          <Text style={[styles.langLabel, { color: colors.textDim }]}>
            {langLabel}
          </Text>
        ) : null}
        <TouchableOpacity onPress={copy} hitSlop={{ top: 4, bottom: 4, left: 4, right: 4 }}>
          <Ionicons name="copy-outline" size={15} color={colors.textDim} />
        </TouchableOpacity>
      </View>
      <ScrollView
        style={styles.codeScroll}
        nestedScrollEnabled
        showsVerticalScrollIndicator={totalLines > 30}
        scrollEventThrottle={32}
        onScroll={onScroll}
      >
        {needsWindowing && topSpacerHeight > 0 ? (
          <View style={{ height: topSpacerHeight }} />
        ) : null}
        {tokenizedLines.slice(startIdx, endIdx).map((tokens, lineIdx) => (
          <Text
            key={startIdx + lineIdx}
            style={styles.line}
          >
            {tokens.map((tok, tokIdx) => (
              <Text
                key={tokIdx}
                style={{
                  color: colors.syntax[tok.type],
                  fontFamily: 'monospace',
                  fontSize: 13,
                  lineHeight: LINE_HEIGHT,
                }}
              >
                {tok.text}
              </Text>
            ))}
            {'\n'}
          </Text>
        ))}
        {needsWindowing && bottomSpacerHeight > 0 ? (
          <View style={{ height: bottomSpacerHeight }} />
        ) : null}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 10,
    padding: 12,
    marginVertical: 5,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  langLabel: {
    fontFamily: 'monospace',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    flex: 1,
  },
  line: {
    fontFamily: 'monospace',
    fontSize: 13,
    lineHeight: 19,
  },
  codeScroll: {
    maxHeight: 400,
  },
});