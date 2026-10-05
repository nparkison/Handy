import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { listen } from "@tauri-apps/api/event";
import {
  commands,
  type SilentMicAlert as SilentMicAlertState,
} from "@/bindings";
import { Alert } from "../ui/Alert";
import { Button } from "../ui/Button";

/**
 * Persistent dead-air alert for General > Sound: shown after several silent
 * recordings in a row from the same microphone. The backend clears it on the
 * next recording with speech or a device change and emits
 * `silent-mic-alert-changed`.
 */
export const SilentMicAlert: React.FC = () => {
  const { t } = useTranslation();
  const [alert, setAlert] = useState<SilentMicAlertState | null>(null);

  useEffect(() => {
    let active = true;
    // An event is newer than the initial read, which may resolve after it.
    let eventSeen = false;
    commands
      .getSilentMicAlert()
      .then((current) => {
        if (active && !eventSeen) setAlert(current);
      })
      .catch((e) => console.warn("Failed to read silent-mic alert:", e));
    const unlisten = listen<SilentMicAlertState | null>(
      "silent-mic-alert-changed",
      (event) => {
        eventSeen = true;
        setAlert(event.payload);
      },
    );
    return () => {
      active = false;
      unlisten.then((fn) => fn());
    };
  }, []);

  if (!alert) return null;

  const dismiss = async () => {
    setAlert(null);
    try {
      await commands.dismissSilentMicAlert();
    } catch (e) {
      console.warn("Failed to dismiss silent-mic alert:", e);
    }
  };

  return (
    <div role="status">
      <Alert variant="warning" contained>
        <span className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <span>
            {t("settings.sound.silentMicAlert.message", {
              count: alert.count,
              mic: alert.mic,
            })}
          </span>
          <Button variant="warning" size="sm" onClick={dismiss}>
            {t("settings.sound.silentMicAlert.dismiss")}
          </Button>
        </span>
      </Alert>
    </div>
  );
};
