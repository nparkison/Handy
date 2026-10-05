import React from "react";
import { useTranslation } from "react-i18next";
import type { AppSettings } from "@/bindings";
import { Alert } from "../../ui/Alert";
import { Dropdown } from "../../ui/Dropdown";
import { SettingContainer } from "../../ui/SettingContainer";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { useSettings } from "../../../hooks/useSettings";

/** Providers that work without an API key. */
const KEYLESS_PROVIDER_IDS = ["custom", "apple_intelligence"];

const DEFAULT_TIMEOUT_MS = 1500;

/**
 * True when cleanup can actually run: a usable prompt exists (the selected
 * one, or the first non-empty one when none is selected), a provider is
 * selected, it has a model, and it has an API key (unless the provider works
 * without one). Mirrors the backend's checks before it sends a
 * post-processing request.
 */
export const isCleanupConfigured = (settings: AppSettings | null): boolean => {
  const prompts = settings?.post_process_prompts ?? [];
  const selected = prompts.find(
    (p) => p.id === settings?.post_process_selected_prompt_id,
  );
  const prompt = selected ?? prompts.find((p) => p.prompt.trim().length > 0);
  if (!prompt || !prompt.prompt.trim()) return false;
  const providerId = settings?.post_process_provider_id;
  if (!providerId) return false;
  const provider = settings?.post_process_providers?.find(
    (p) => p.id === providerId,
  );
  if (!provider) return false;
  const model = settings?.post_process_models?.[providerId] ?? "";
  if (!model.trim()) return false;
  if (KEYLESS_PROVIDER_IDS.includes(providerId)) return true;
  const apiKey = settings?.post_process_api_keys?.[providerId] ?? "";
  return apiKey.trim().length > 0;
};

interface CleanupRowProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const CleanUpEveryDictation: React.FC<CleanupRowProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { settings, getSetting, updateSetting, isUpdating } = useSettings();

  const enabled = getSetting("post_process_every_dictation") ?? false;
  const configured = isCleanupConfigured(settings);

  return (
    <>
      <ToggleSwitch
        checked={enabled}
        onChange={(value) =>
          updateSetting("post_process_every_dictation", value)
        }
        isUpdating={isUpdating("post_process_every_dictation")}
        // Still allow turning it off if the provider stopped being configured.
        disabled={!configured && !enabled}
        label={t("settings.postProcessing.everyDictation.label")}
        description={t("settings.postProcessing.everyDictation.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
      {!configured && (
        <Alert variant="info" contained>
          {t("settings.postProcessing.everyDictation.needsProvider")}
        </Alert>
      )}
    </>
  );
};

const TIMEOUT_OPTIONS: { ms: number; labelKey: string }[] = [
  { ms: 1000, labelKey: "settings.postProcessing.timeout.options.sec1" },
  { ms: 1500, labelKey: "settings.postProcessing.timeout.options.sec1_5" },
  { ms: 2000, labelKey: "settings.postProcessing.timeout.options.sec2" },
  { ms: 3000, labelKey: "settings.postProcessing.timeout.options.sec3" },
  { ms: 5000, labelKey: "settings.postProcessing.timeout.options.sec5" },
  { ms: 0, labelKey: "settings.postProcessing.timeout.options.noLimit" },
];

export const CleanupTimeLimit: React.FC<CleanupRowProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const current = getSetting("post_process_timeout_ms") ?? DEFAULT_TIMEOUT_MS;

  const options = TIMEOUT_OPTIONS.map(({ ms, labelKey }) => ({
    value: String(ms),
    label: t(labelKey),
  }));
  // Keep a value set outside the presets (e.g. by editing settings) visible.
  if (!TIMEOUT_OPTIONS.some(({ ms }) => ms === current)) {
    options.push({
      value: String(current),
      label: t("settings.postProcessing.timeout.options.custom", {
        seconds: current / 1000,
      }),
    });
  }

  return (
    <SettingContainer
      title={t("settings.postProcessing.timeout.title")}
      description={t("settings.postProcessing.timeout.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
    >
      <Dropdown
        options={options}
        selectedValue={String(current)}
        onSelect={(value) =>
          updateSetting("post_process_timeout_ms", Number(value))
        }
        disabled={isUpdating("post_process_timeout_ms")}
      />
    </SettingContainer>
  );
};
