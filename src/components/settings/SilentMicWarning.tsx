import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface SilentMicWarningProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/** Dead-air guard toggle: skip silent recordings and name the mic to check. */
export const SilentMicWarning: React.FC<SilentMicWarningProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("silent_mic_warning") ?? true;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(value) => updateSetting("silent_mic_warning", value)}
        isUpdating={isUpdating("silent_mic_warning")}
        label={t("settings.sound.silentMicWarning.label")}
        description={t("settings.sound.silentMicWarning.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
