import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface ShowRecentDictationsInTrayProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const ShowRecentDictationsInTray: React.FC<ShowRecentDictationsInTrayProps> =
  React.memo(({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("show_recent_dictations_in_tray") ?? true;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(value) =>
          updateSetting("show_recent_dictations_in_tray", value)
        }
        isUpdating={isUpdating("show_recent_dictations_in_tray")}
        label={t("settings.advanced.showRecentDictationsInTray.label")}
        description={t(
          "settings.advanced.showRecentDictationsInTray.description",
        )}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  });
