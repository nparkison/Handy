import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ask } from "@tauri-apps/plugin-dialog";
import { useSettings } from "../../hooks/useSettings";
import { useSettingsStore } from "../../stores/settingsStore";
import { Input } from "../ui/Input";
import { SettingContainer } from "../ui/SettingContainer";

const MIN_HISTORY_LIMIT = 0;
const MAX_HISTORY_LIMIT = 1000;

interface HistoryLimitProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

export const HistoryLimit: React.FC<HistoryLimitProps> = ({
  descriptionMode = "inline",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const historyLimit = getSetting("history_limit") ?? 200;
  const retention = getSetting("recording_retention_period");
  // Typing is only a draft: saving prunes History right away, so committing
  // each keystroke (e.g. the "1" of "150") would delete almost everything.
  const [draft, setDraft] = useState(String(historyLimit));

  useEffect(() => {
    setDraft(String(historyLimit));
  }, [historyLimit]);

  const reset = () => setDraft(String(historyLimit));

  const commit = async () => {
    const trimmed = draft.trim();
    const parsed = Number(trimmed);
    if (trimmed === "" || !Number.isInteger(parsed)) {
      reset();
      return;
    }
    const value = Math.min(
      MAX_HISTORY_LIMIT,
      Math.max(MIN_HISTORY_LIMIT, parsed),
    );
    if (value === historyLimit) {
      reset();
      return;
    }
    // The limit only prunes under the count-based retention setting.
    const prunes =
      (retention ?? "preserve_limit") === "preserve_limit" &&
      value < historyLimit;
    if (prunes) {
      let confirmed = false;
      try {
        confirmed = await ask(
          t("settings.debug.historyLimit.lowerConfirm", { count: value }),
          {
            title: t("settings.debug.historyLimit.lowerTitle"),
            kind: "warning",
          },
        );
      } catch (error) {
        console.error("Failed to confirm history limit change:", error);
      }
      if (!confirmed) {
        reset();
        return;
      }
    }
    await updateSetting("history_limit", value);
    // A failed save is rolled back in the store; show what is actually saved.
    const saved = useSettingsStore.getState().settings?.history_limit;
    setDraft(String(saved ?? historyLimit));
  };

  const handleBlur = () => {
    // Blur also fires when the whole window loses focus or is hidden (alt-tab,
    // close to tray). Don't commit then: a confirm dialog would pop up over
    // another app or a hidden window. The draft stays and commits on the next
    // real blur.
    if (document.hidden || !document.hasFocus()) return;
    void commit();
  };

  return (
    <SettingContainer
      title={t("settings.debug.historyLimit.title")}
      description={t("settings.debug.historyLimit.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
      layout="horizontal"
    >
      <div className="flex items-center space-x-2">
        <Input
          type="number"
          min={MIN_HISTORY_LIMIT}
          max={MAX_HISTORY_LIMIT}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={handleBlur}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.currentTarget.blur();
            } else if (event.key === "Escape") {
              reset();
            }
          }}
          disabled={isUpdating("history_limit")}
          className="w-20"
        />
        <span className="text-sm text-text">
          {t("settings.debug.historyLimit.entries")}
        </span>
      </div>
    </SettingContainer>
  );
};
