import React, { useEffect, useId, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ArrowDown, ArrowUp, Plus, Trash2 } from "lucide-react";
import { commands } from "@/bindings";
import type { AppContextMode, AppRule, AppRuleMatch } from "@/bindings";
import { Dropdown } from "../../ui/Dropdown";
import { SettingContainer } from "../../ui/SettingContainer";
import { Button } from "../../ui/Button";
import { Input } from "../../ui/Input";
import { useSettings } from "../../../hooks/useSettings";
import { useSettingsStore } from "../../../stores/settingsStore";

/** Same cap as the backend (`MAX_APP_RULES` in shortcut/mod.rs). */
const MAX_APP_RULES = 100;

const DEFAULT_MODE: AppContextMode = "app_and_title";

const newRuleId = (): string => {
  try {
    return crypto.randomUUID();
  } catch {
    return `rule-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }
};

export const ShareAppInfo: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const mode = getSetting("app_context_mode") ?? DEFAULT_MODE;

  const options: { value: AppContextMode; label: string }[] = [
    {
      value: "off",
      label: t("settings.postProcessing.context.shareAppInfo.options.off"),
    },
    {
      value: "app_name",
      label: t("settings.postProcessing.context.shareAppInfo.options.appName"),
    },
    {
      value: "app_and_title",
      label: t(
        "settings.postProcessing.context.shareAppInfo.options.appAndTitle",
      ),
    },
  ];

  return (
    <SettingContainer
      title={t("settings.postProcessing.context.shareAppInfo.title")}
      description={t(
        "settings.postProcessing.context.shareAppInfo.description",
      )}
      descriptionMode="tooltip"
      grouped={true}
    >
      <Dropdown
        options={options}
        selectedValue={mode}
        onSelect={(value) =>
          updateSetting("app_context_mode", value as AppContextMode)
        }
        disabled={isUpdating("app_context_mode")}
      />
    </SettingContainer>
  );
};

interface RuleRowProps {
  rule: AppRule;
  index: number;
  count: number;
  promptOptions: { value: string; label: string }[];
  /** Why screenshots can't be used right now, if they can't. */
  screenshotBlocked: "noVision" | "shareOff" | null;
  autoFocus: boolean;
  onChange: (patch: Partial<AppRule>) => void;
  onMove: (delta: -1 | 1) => void;
  onDelete: () => void;
}

const RuleRow: React.FC<RuleRowProps> = ({
  rule,
  index,
  count,
  promptOptions,
  screenshotBlocked,
  autoFocus,
  onChange,
  onMove,
  onDelete,
}) => {
  const { t } = useTranslation();
  const baseId = useId();
  const pattern = rule.pattern ?? "";
  const matchOn: AppRuleMatch = rule.match_on ?? "app";
  const [draft, setDraft] = useState(pattern);
  const [editing, setEditing] = useState(false);
  // Escape discards the edit instead of committing it on blur.
  const discardRef = useRef(false);

  // Follow outside changes (reorder, reload) while not editing.
  useEffect(() => {
    if (!editing) setDraft(pattern);
  }, [pattern, editing]);

  const commitDraft = () => {
    setEditing(false);
    if (discardRef.current) {
      discardRef.current = false;
      setDraft(pattern);
      return;
    }
    const trimmed = draft.trim();
    if (trimmed !== pattern) onChange({ pattern: trimmed });
  };

  const promptMissing = !promptOptions.some(
    (option) => option.value === rule.prompt_id,
  );
  const screenshot = rule.screenshot ?? false;
  const canScreenshot = screenshotBlocked === null;
  const ruleLabel = t("settings.postProcessing.context.rules.ruleLabel", {
    number: index + 1,
  });

  return (
    <li
      className="flex flex-col gap-2 rounded-md border border-mid-gray/30 p-3"
      aria-label={ruleLabel}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-text/60">{ruleLabel}</span>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onMove(-1)}
            disabled={index === 0}
            aria-label={t("settings.postProcessing.context.rules.moveUp")}
            title={t("settings.postProcessing.context.rules.moveUp")}
          >
            <ArrowUp width={14} height={14} aria-hidden="true" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onMove(1)}
            disabled={index === count - 1}
            aria-label={t("settings.postProcessing.context.rules.moveDown")}
            title={t("settings.postProcessing.context.rules.moveDown")}
          >
            <ArrowDown width={14} height={14} aria-hidden="true" />
          </Button>
          <Button
            variant="danger-ghost"
            size="sm"
            onClick={onDelete}
            aria-label={t("settings.postProcessing.context.rules.delete")}
            title={t("settings.postProcessing.context.rules.delete")}
          >
            <Trash2 width={14} height={14} aria-hidden="true" />
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm w-16 shrink-0">
          {t("settings.postProcessing.context.rules.match")}
        </span>
        <Dropdown
          options={[
            {
              value: "app",
              label: t("settings.postProcessing.context.rules.matchApp"),
            },
            {
              value: "title",
              label: t("settings.postProcessing.context.rules.matchTitle"),
            },
          ]}
          selectedValue={matchOn}
          onSelect={(value) => onChange({ match_on: value as AppRuleMatch })}
        />
        <Input
          autoFocus={autoFocus}
          variant="compact"
          className="flex-1 min-w-40"
          value={draft}
          aria-label={t("settings.postProcessing.context.rules.pattern")}
          placeholder={
            matchOn === "app"
              ? t("settings.postProcessing.context.rules.appPlaceholder")
              : t("settings.postProcessing.context.rules.titlePlaceholder")
          }
          onFocus={() => setEditing(true)}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commitDraft}
          onKeyDown={(event) => {
            if (event.key === "Enter") event.currentTarget.blur();
            if (event.key === "Escape") {
              discardRef.current = true;
              event.currentTarget.blur();
            }
          }}
        />
      </div>
      {draft.trim().length === 0 && (
        <p className="text-xs text-text/60">
          {t("settings.postProcessing.context.rules.emptyPattern")}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm w-16 shrink-0">
          {t("settings.postProcessing.context.rules.prompt")}
        </span>
        <Dropdown
          options={promptOptions}
          selectedValue={promptMissing ? null : (rule.prompt_id ?? null)}
          placeholder={t("settings.postProcessing.context.rules.promptMissing")}
          onSelect={(value) => onChange({ prompt_id: value })}
        />
      </div>
      {promptMissing && (
        <p className="text-xs text-orange-600 dark:text-orange-400">
          {t("settings.postProcessing.context.rules.promptMissingNote")}
        </p>
      )}

      <div className="flex flex-col gap-1">
        <label
          htmlFor={`${baseId}-screenshot`}
          className={`flex items-center gap-2 text-sm ${
            canScreenshot ? "cursor-pointer" : "text-text/50 cursor-default"
          }`}
        >
          <input
            id={`${baseId}-screenshot`}
            type="checkbox"
            // Shown unchecked while screenshots can't be taken, so the
            // checkbox never claims something that won't happen.
            checked={screenshot && canScreenshot}
            disabled={!canScreenshot}
            aria-describedby={`${baseId}-screenshot-note`}
            onChange={(event) => onChange({ screenshot: event.target.checked })}
            className="accent-logo-primary"
          />
          {t("settings.postProcessing.context.rules.screenshot")}
        </label>
        <p id={`${baseId}-screenshot-note`} className="text-xs text-text/60">
          {screenshotBlocked === "shareOff"
            ? t("settings.postProcessing.context.rules.screenshotShareOff")
            : screenshotBlocked === "noVision"
              ? t("settings.postProcessing.context.rules.screenshotNoVision")
              : screenshot
                ? t("settings.postProcessing.context.rules.screenshotLatency")
                : null}
        </p>
      </div>
    </li>
  );
};

export const AppRules: React.FC = () => {
  const { t } = useTranslation();
  const { settings, getSetting, updateSetting } = useSettings();
  const rules = getSetting("app_rules") ?? [];
  const prompts = settings?.post_process_prompts ?? [];
  const provider = settings?.post_process_providers?.find(
    (p) => p.id === settings?.post_process_provider_id,
  );
  const supportsVision = provider?.supports_vision ?? false;
  const shareOff = (getSetting("app_context_mode") ?? DEFAULT_MODE) === "off";
  const screenshotBlocked = shareOff
    ? "shareOff"
    : !supportsVision
      ? "noVision"
      : null;
  const showMovedNote = getSetting("show_screen_context_moved_note") ?? false;
  const [picking, setPicking] = useState(false);
  const [openingPicker, setOpeningPicker] = useState(false);
  const lastAddRef = useRef(0);
  const [recentApps, setRecentApps] = useState<string[]>([]);
  const [focusRuleId, setFocusRuleId] = useState<string | null>(null);

  const promptOptions = prompts.map((prompt) => ({
    value: prompt.id,
    label: prompt.name,
  }));
  const defaultPromptId =
    settings?.post_process_selected_prompt_id ?? prompts[0]?.id ?? "";

  const save = (next: AppRule[]) => updateSetting("app_rules", next);
  // Always edit the newest list: a save or reload may have landed since
  // this render (e.g. while the recent-apps lookup was awaited).
  const latestRules = (): AppRule[] =>
    useSettingsStore.getState().settings?.app_rules ?? [];

  const atLimit = rules.length >= MAX_APP_RULES;

  const addRule = (pattern: string) => {
    // The backend refuses more than MAX_APP_RULES rules.
    if (latestRules().length >= MAX_APP_RULES) {
      setPicking(false);
      return;
    }
    // A double click must not add the rule twice.
    const now = Date.now();
    if (now - lastAddRef.current < 400) return;
    lastAddRef.current = now;
    const rule: AppRule = {
      id: newRuleId(),
      match_on: "app",
      pattern,
      prompt_id: defaultPromptId,
      screenshot: false,
    };
    setPicking(false);
    setFocusRuleId(pattern ? null : rule.id);
    save([...latestRules(), rule]);
  };

  const openPicker = async () => {
    if (openingPicker) return;
    setOpeningPicker(true);
    let apps: string[] = [];
    try {
      const result = await commands.getRecentContextApps();
      if (result.status === "ok") apps = result.data;
    } catch (error) {
      console.warn("Failed to load recent apps:", error);
    } finally {
      setOpeningPicker(false);
    }
    const covered = new Set(
      latestRules()
        .filter((rule) => (rule.match_on ?? "app") === "app")
        .map((rule) => (rule.pattern ?? "").toLowerCase()),
    );
    const candidates = apps.filter((app) => !covered.has(app.toLowerCase()));
    if (candidates.length === 0) {
      addRule("");
      return;
    }
    setRecentApps(candidates);
    setPicking(true);
  };

  const updateRule = (index: number, patch: Partial<AppRule>) => {
    save(
      latestRules().map((rule, i) =>
        i === index ? { ...rule, ...patch } : rule,
      ),
    );
  };

  const moveRule = (index: number, delta: -1 | 1) => {
    const current = latestRules();
    const target = index + delta;
    if (target < 0 || target >= current.length) return;
    const next = [...current];
    [next[index], next[target]] = [next[target], next[index]];
    save(next);
  };

  const deleteRule = (index: number) => {
    save(latestRules().filter((_, i) => i !== index));
  };

  return (
    <SettingContainer
      title={t("settings.postProcessing.context.rules.title")}
      description={t("settings.postProcessing.context.rules.description")}
      descriptionMode="tooltip"
      grouped={true}
      layout="stacked"
    >
      <div className="flex flex-col gap-3">
        {showMovedNote && (
          <div className="flex items-start justify-between gap-2 rounded-md border border-logo-primary/40 bg-logo-primary/10 p-3">
            <p className="text-sm">
              {t("settings.postProcessing.context.screenContextMoved")}
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                updateSetting("show_screen_context_moved_note", false)
              }
            >
              {t("settings.postProcessing.context.dismiss")}
            </Button>
          </div>
        )}
        {rules.length === 0 ? (
          <p className="text-sm text-text/60">
            {t("settings.postProcessing.context.rules.empty")}
          </p>
        ) : (
          <ol className="flex flex-col gap-2">
            {rules.map((rule, index) => (
              <RuleRow
                key={rule.id}
                rule={rule}
                index={index}
                count={rules.length}
                promptOptions={promptOptions}
                screenshotBlocked={screenshotBlocked}
                autoFocus={focusRuleId === rule.id}
                onChange={(patch) => updateRule(index, patch)}
                onMove={(delta) => moveRule(index, delta)}
                onDelete={() => deleteRule(index)}
              />
            ))}
          </ol>
        )}

        {picking && !atLimit ? (
          <div className="flex flex-col gap-2 rounded-md border border-mid-gray/30 p-3">
            <span className="text-xs font-medium text-text/60">
              {t("settings.postProcessing.context.rules.recentApps")}
            </span>
            <div className="flex flex-wrap gap-2">
              {recentApps.map((app) => (
                <Button
                  key={app}
                  variant="secondary"
                  size="sm"
                  onClick={() => addRule(app)}
                >
                  {app}
                </Button>
              ))}
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="primary-soft"
                size="sm"
                onClick={() => addRule("")}
              >
                {t("settings.postProcessing.context.rules.typeAppName")}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPicking(false)}
              >
                {t("settings.postProcessing.context.rules.cancel")}
              </Button>
            </div>
          </div>
        ) : (
          <Button
            variant="secondary"
            size="sm"
            className="self-start flex items-center gap-1"
            onClick={openPicker}
            disabled={openingPicker || atLimit}
          >
            <Plus width={14} height={14} aria-hidden="true" />
            {t("settings.postProcessing.context.rules.add")}
          </Button>
        )}
        {atLimit && (
          <p className="text-xs text-text/60">
            {t("settings.postProcessing.context.rules.limitReached", {
              max: MAX_APP_RULES,
            })}
          </p>
        )}
      </div>
    </SettingContainer>
  );
};
