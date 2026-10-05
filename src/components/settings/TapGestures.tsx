import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface TapGesturesProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Tap / double-tap gestures on the transcribe shortcut (paste last / swap
 * last). Only meaningful in Hold (push-to-talk) mode; in Auto and Toggle the
 * row is disabled and the stored value is preserved.
 */
export const TapGestures: React.FC<TapGesturesProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("tap_gestures_enabled") ?? false;
    const isHoldMode = getSetting("shortcut_activation") === "push_to_talk";

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(value) => updateSetting("tap_gestures_enabled", value)}
        isUpdating={isUpdating("tap_gestures_enabled")}
        disabled={!isHoldMode}
        label={t("settings.general.tapGestures.label")}
        description={
          isHoldMode
            ? t("settings.general.tapGestures.description")
            : t("settings.general.tapGestures.unavailable")
        }
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
