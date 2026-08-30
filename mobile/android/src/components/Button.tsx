import React from 'react';
import {
  TouchableOpacity,
  Text as RNText,
  StyleSheet,
  TextStyle,
  ViewStyle,
  ActivityIndicator,
} from 'react-native';
import { useTheme } from '../theme/ThemeContext';

export type ButtonProps = {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  disabled?: boolean;
  loading?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
  /** Round-32b F11: expose a11y state (e.g. {disabled: true}) to screen
   * readers while a conflict action is in progress. */
  accessibilityState?: { disabled?: boolean; busy?: boolean; checked?: boolean };
};

export function Button({
  title,
  onPress,
  variant = 'primary',
  disabled,
  loading,
  style,
  textStyle,
  accessibilityState,
}: ButtonProps) {
  const { colors, spacing, radii, typography } = useTheme();

  const baseStyle: ViewStyle = {
    minHeight: 44,
    borderRadius: radii.sm,
    paddingHorizontal: spacing.base,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    borderWidth: StyleSheet.hairlineWidth,
    opacity: disabled ? 0.42 : 1,
  };

  const variantStyles: Record<string, ViewStyle> = {
    primary: { backgroundColor: colors.accentStrong, borderColor: colors.accentStrong },
    secondary: { backgroundColor: 'transparent', borderColor: colors.hairline },
    ghost: { backgroundColor: 'transparent', borderColor: 'transparent' },
    danger: { backgroundColor: colors.error, borderColor: colors.error },
  };

  const textColor = variant === 'primary' || variant === 'danger'
    ? colors.accentText
    : variant === 'ghost'
      ? colors.textDim
      : colors.textSoft;
  const fontSpec = typography.sm;

  return (
    <TouchableOpacity
      accessibilityRole="button"
      accessibilityState={accessibilityState ?? (disabled || loading ? { disabled: true } : undefined)}
      onPress={onPress}
      disabled={disabled || loading}
      style={[baseStyle, variantStyles[variant], style]}
      activeOpacity={0.72}
    >
      {loading && <ActivityIndicator size="small" color={textColor} style={{ marginRight: 8 }} />}
      <RNText style={[{ color: textColor, fontSize: fontSpec.fontSize, fontWeight: '600' }, textStyle]}>
        {title}
      </RNText>
    </TouchableOpacity>
  );
}
