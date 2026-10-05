// Payload of the `overlay-notice` event (Rust: overlay_notice::OverlayNoticePayload).
// Emitted as a plain event, so it is not part of the generated bindings.

export interface NoticeText {
  key: string;
  params: Record<string, string>;
}

export interface OverlayNotice {
  id: number;
  kind: "info" | "warning";
  message: NoticeText;
  action: NoticeText | null;
  duration_ms: number;
  urgent: boolean;
}

/** Param values longer than this are truncated on screen (full text in the tooltip). */
export const NOTICE_PARAM_MAX_CHARS = 28;

export const truncateParam = (
  value: string,
  max = NOTICE_PARAM_MAX_CHARS,
): string => {
  const chars = Array.from(value);
  return chars.length > max ? `${chars.slice(0, max - 1).join("")}…` : value;
};

export const truncateParams = (
  params: Record<string, string>,
): Record<string, string> =>
  Object.fromEntries(
    Object.entries(params).map(([name, value]) => [name, truncateParam(value)]),
  );
