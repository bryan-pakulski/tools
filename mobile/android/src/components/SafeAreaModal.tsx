import React from 'react';
import { Modal, StyleSheet } from 'react-native';
import type { ModalProps, StyleProp, ViewStyle } from 'react-native';
import {
  SafeAreaProvider,
  SafeAreaView,
  initialWindowMetrics,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';
import type { Edge } from 'react-native-safe-area-context';
import { AtmosphericBackground } from './AtmosphericBackground';

export type SafeAreaModalProps = Omit<ModalProps, 'children'> & {
  children: React.ReactNode;
  edges?: Edge[];
  containerStyle?: StyleProp<ViewStyle>;
};

/**
 * React Native modals render in their own native window. Full-screen modal
 * workflows receive the same atmospheric product background as the app;
 * transparent overlays keep the underlying app visible instead.
 */
/** Modal-local insets: navigation bar must not bleed under modal content. */
const navigationBarTranslucent = false;

function ModalInsets({ edges }: { edges: Edge[] }) {
  const insets = useSafeAreaInsets();
  void insets; // consumed by SafeAreaView inside the provider below
  return null;
}

export function SafeAreaModal({
  children,
  edges = ['top', 'bottom'],
  containerStyle,
  statusBarTranslucent = false,
  transparent = false,
  ...modalProps
}: SafeAreaModalProps) {
  const safeArea = (
    <SafeAreaProvider initialMetrics={initialWindowMetrics} style={styles.provider}>
      <ModalInsets edges={edges} />
      <SafeAreaView edges={edges} style={[styles.container, containerStyle]}>
        {children}
      </SafeAreaView>
    </SafeAreaProvider>
  );

  return (
    <Modal
      {...modalProps}
      transparent={transparent}
      statusBarTranslucent={statusBarTranslucent}
    >
      {transparent ? safeArea : <AtmosphericBackground>{safeArea}</AtmosphericBackground>}
    </Modal>
  );
}

const styles = StyleSheet.create({
  provider: { flex: 1 },
  container: { flex: 1 },
});
