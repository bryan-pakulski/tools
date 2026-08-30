// Mock AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(() => Promise.resolve(null)),
  setItem: jest.fn(() => Promise.resolve()),
  removeItem: jest.fn(() => Promise.resolve()),
}));
// RN Modal renders in a separate native window; under jest (test env) its
// children are not mounted, so BottomSheet/BottomSheet-backed tests see an
// empty tree. Mock Modal to a plain View that always renders its children.
jest.mock('react-native/Libraries/Modal/Modal', () => {
  const React = require('react');
  const { View } = require('react-native');
  return ({ children, testID, visible }: { children?: React.ReactNode; testID?: string; visible?: boolean }) =>
    visible === false ? null : React.createElement(View, { testID }, children);
});

// react-native-safe-area-context native components render nothing under jest;
// replace with pass-through views so SafeAreaModal/SafeAreaView trees mount.
jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const { View } = require('react-native');
  const PassThrough = ({ children, ...rest }: { children?: React.ReactNode; [k: string]: unknown }) =>
    React.createElement(View, rest, children);
  return {
    SafeAreaProvider: PassThrough,
    SafeAreaView: PassThrough,
    SafeAreaConsumer: ({ children }: { children?: React.ReactNode }) => children ?? null,
    SafeAreaInsetsContext: {
      Provider: ({ children }: { children?: React.ReactNode }) => children ?? null,
      Consumer: ({ children }: { children?: (v: unknown) => React.ReactNode }) => children?.({ top: 0, bottom: 0, left: 0, right: 0 }) ?? null,
    },
    useSafeAreaInsets: () => ({ top: 0, bottom: 0, left: 0, right: 0 }),
    useSafeAreaFrame: () => ({ x: 0, y: 0, width: 390, height: 844 }),
    initialWindowMetrics: {
      frame: { x: 0, y: 0, width: 0, height: 0 },
      insets: { top: 0, bottom: 0, left: 0, right: 0 },
    },
    withSafeAreaInsets: (C: unknown) => C,
  };
});
