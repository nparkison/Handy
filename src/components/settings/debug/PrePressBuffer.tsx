import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../../ui/Slider";
import { useSettings } from "../../../hooks/useSettings";

interface PrePressBufferProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

export const PrePressBuffer: React.FC<PrePressBufferProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { settings, updateSetting, resetSetting, isUpdating } = useSettings();

  // Pre-roll only has audio to keep while the microphone stream is already
  // open; it never opens the microphone on its own.
  const micIsWarm =
    (settings?.always_on_microphone ?? false) ||
    (settings?.lazy_stream_close ?? false);

  return (
    <Slider
      value={settings?.pre_roll_ms ?? 300}
      onChange={(value) => updateSetting("pre_roll_ms", value)}
      onReset={() => resetSetting("pre_roll_ms")}
      isResetting={isUpdating("pre_roll_ms")}
      disabled={!micIsWarm}
      min={0}
      max={1000}
      step={50}
      label={t("settings.debug.prePressBuffer.title")}
      description={
        micIsWarm
          ? t("settings.debug.prePressBuffer.description")
          : t("settings.debug.prePressBuffer.needsWarmMic")
      }
      descriptionMode={micIsWarm ? descriptionMode : "inline"}
      grouped={grouped}
      formatValue={(v) => `${v}ms`}
    />
  );
};
