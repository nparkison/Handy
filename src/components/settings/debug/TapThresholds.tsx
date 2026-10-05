import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../../ui/Slider";
import { useSettings } from "../../../hooks/useSettings";

interface TapThresholdProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

/**
 * Tap gestures: presses shorter than this (with no speech) count as a tap.
 */
export const TapMaxDuration: React.FC<TapThresholdProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { settings, updateSetting, resetSetting, isUpdating } = useSettings();

  return (
    <Slider
      value={settings?.tap_max_duration_ms ?? 200}
      onChange={(value) => updateSetting("tap_max_duration_ms", value)}
      onReset={() => resetSetting("tap_max_duration_ms")}
      isResetting={isUpdating("tap_max_duration_ms")}
      min={100}
      max={400}
      step={10}
      label={t("settings.debug.tapMaxDuration.title")}
      description={t("settings.debug.tapMaxDuration.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
      formatValue={(v) => `${v}ms`}
    />
  );
};

/**
 * Tap gestures: how soon after the first tap's release a second tap must
 * start to count as a double-tap.
 */
export const DoubleTapWindow: React.FC<TapThresholdProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { settings, updateSetting, resetSetting, isUpdating } = useSettings();

  return (
    <Slider
      value={settings?.double_tap_window_ms ?? 250}
      onChange={(value) => updateSetting("double_tap_window_ms", value)}
      onReset={() => resetSetting("double_tap_window_ms")}
      isResetting={isUpdating("double_tap_window_ms")}
      min={150}
      max={600}
      step={25}
      label={t("settings.debug.doubleTapWindow.title")}
      description={t("settings.debug.doubleTapWindow.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
      formatValue={(v) => `${v}ms`}
    />
  );
};
