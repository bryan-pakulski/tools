import React, { useCallback, useState } from 'react';
import { Pressable, RefreshControl, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { openExternalUrl } from '../api/urlSafety';
import { Badge, Card, EmptyState, ErrorState, ModeWorkspaceHeader, Skeleton, Text, useModeWorkspaceView } from '../components';
import { researchApi, ResearchState } from '../api/research';
import { spacing } from '../theme/tokens';

export function ResearchScreen() {
  const { colors } = useTheme();
  const [state, setState] = useState<ResearchState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [expandedSource, setExpandedSource] = useState<number | null>(null);
  const workspaceView = useModeWorkspaceView(state?.workspace);

  const load = useCallback(async () => {
    try {
      setError(null);
      setState(await researchApi.getState());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(useCallback(() => { setLoading(true); load(); }, [load]));

  if (loading) return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <View style={{ padding: spacing.base }}><Skeleton height={280} /><Skeleton height={90} style={{ marginTop: spacing.sm }} /></View>
    </SafeAreaView>
  );
  if (error) return <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}><ErrorState message={error} onRetry={load} /></SafeAreaView>;
  if (!state || (!state.active && !state.sources.length && !state.findings.length)) return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}><EmptyState title="No research session" message="Start a research conversation to build a source-backed evidence desk." /></SafeAreaView>
  );

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <ScrollView
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => { setRefreshing(true); load(); }} />}
        contentContainerStyle={{ padding: spacing.base, paddingBottom: spacing['2xl'] }}
      >
        <ModeWorkspaceHeader workspace={state.workspace} selectedView={workspaceView.selectedView} onSelectView={workspaceView.selectView} />

        {workspaceView.shows('claims') && (
          <View style={{ marginBottom: spacing.md }}>
            <SectionTitle title="Claims" count={state.findings.length} />
            {state.findings.length === 0 ? <QuietEmpty text="No claims captured yet." /> : state.findings.map(finding => (
              <Card key={finding.id} style={{ marginBottom: spacing.sm }}>
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                  <Badge label={finding.record_type === 'claim' ? 'research claim' : 'legacy note'} variant={finding.record_type === 'claim' ? 'accent' : 'neutral'} />
                  <Badge label={finding.source ? 'source linked' : 'evidence gap'} variant={finding.source ? 'success' : 'warning'} />
                  {finding.tags.slice(0, 3).map(tag => <Badge key={tag} label={tag} variant="neutral" />)}
                </View>
                <Text variant="sm" style={{ color: colors.text, marginTop: spacing.sm, lineHeight: 20 }}>{finding.content}</Text>
                <Text variant="xs" style={{ color: colors.textDim, marginTop: 6, fontFamily: 'monospace' }}>
                  {finding.source ? `↳ ${finding.source}` : 'Accuracy and relevance have not been independently assessed'}
                </Text>
              </Card>
            ))}
          </View>
        )}

        {workspaceView.shows('sources') && (
          <View style={{ marginBottom: spacing.md }}>
            <SectionTitle title="Source ledger" count={state.sources.length} />
            {state.sources.length === 0 ? <QuietEmpty text="No sources cited yet." /> : state.sources.map(source => {
              const expanded = expandedSource === source.id;
              const credibility = Math.round((source.credibility_score || 0) * 100);
              return (
                <Pressable key={source.id} onPress={() => setExpandedSource(expanded ? null : source.id)}>
                  <Card style={{ marginBottom: spacing.sm }}>
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                      <Badge label={source.source_type} variant="neutral" />
                      <Text variant="xs" style={{ marginLeft: 'auto', color: colors.textDim, fontFamily: 'monospace' }}>source quality {credibility}%</Text>
                    </View>
                    <Text variant="sm" style={{ color: colors.text, fontWeight: '600', marginTop: spacing.sm }}>{source.title}</Text>
                    <View style={{ height: 3, marginTop: spacing.sm, borderRadius: 2, overflow: 'hidden', backgroundColor: colors.bgHover }}>
                      <View style={{ height: '100%', width: `${credibility}%`, backgroundColor: colors.accent }} />
                    </View>
                    <Text variant="xs" numberOfLines={expanded ? undefined : 1} style={{ color: colors.accent, marginTop: 7 }} onPress={() => void openExternalUrl(source.url)}>{source.url}</Text>
                    {expanded && (
                      <View style={{ marginTop: spacing.sm, paddingTop: spacing.sm, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.border }}>
                        {!!source.authors.length && <MetaRow label="Authors" value={source.authors.join(', ')} />}
                        <MetaRow label="Published" value={source.date || 'Not recorded'} />
                        <MetaRow label="Accessed" value={source.accessed_date || 'Not recorded'} />
                        <Text variant="xs" style={{ color: colors.textDim, marginTop: spacing.sm, lineHeight: 17 }}>
                          Credibility describes the source. It does not establish the accuracy of every claim.
                        </Text>
                      </View>
                    )}
                  </Card>
                </Pressable>
              );
            })}
          </View>
        )}

        {workspaceView.shows('bibliography') && !!state.bibliography && (
          <View>
            <SectionTitle title="Citation output" count={state.source_count} />
            <Card><Text variant="xs" style={{ color: colors.textSoft, fontFamily: 'monospace', lineHeight: 18 }}>{state.bibliography}</Text></Card>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function SectionTitle({ title, count }: { title: string; count: number }) {
  const { colors } = useTheme();
  return <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: spacing.sm }}><Text variant="xs" style={{ color: colors.textSoft, fontWeight: '700', letterSpacing: .9, textTransform: 'uppercase' }}>{title}</Text><Text variant="xs" style={{ marginLeft: 'auto', color: colors.textDim, fontFamily: 'monospace' }}>{count}</Text></View>;
}

function QuietEmpty({ text }: { text: string }) {
  const { colors } = useTheme();
  return <Text variant="xs" style={{ color: colors.textDim, paddingVertical: spacing.md }}>{text}</Text>;
}

function MetaRow({ label, value }: { label: string; value: string }) {
  const { colors } = useTheme();
  return <View style={{ flexDirection: 'row', gap: spacing.sm, paddingVertical: 3 }}><Text variant="xs" style={{ width: 70, color: colors.textDim }}>{label}</Text><Text variant="xs" style={{ flex: 1, color: colors.textSoft }}>{value}</Text></View>;
}
