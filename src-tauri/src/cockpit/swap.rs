//! Swap-last safety guards (pure logic).
//!
//! An in-place swap deletes the text Handy just pasted by selecting it back
//! with Shift+Left × N and pasting the other version over the selection. It is
//! never done with Ctrl+Z (Word/Docs/Slack merge undo groups, and Ctrl+Z
//! suspends jobs in terminals) or with backspace runs. A wrong delete destroys
//! user text, so every guard must pass; otherwise the other version is only
//! copied to the clipboard.

use std::time::Duration;

/// In-place swaps are only attempted this soon after the paste.
pub const MAX_SWAP_AGE: Duration = Duration::from_secs(120);

/// Selecting back more than this many characters is slow and fragile.
pub const MAX_SELECT_BACK_CHARS: usize = 500;

/// Executables (Windows) and app names (macOS/Linux) that are terminals.
/// Matched case-insensitively against the process file name, with or without
/// ".exe". VS Code's integrated terminal reports Code.exe and cannot be
/// detected here; the "no input since paste" guard is the safety net.
const TERMINALS: &[&str] = &[
    "windowsterminal",
    "openconsole",
    "conhost",
    "cmd",
    "powershell",
    "powershell_ise",
    "pwsh",
    "wezterm-gui",
    "wezterm",
    "alacritty",
    "mintty",
    "putty",
    "kitty",
    "iterm2",
    "terminal",
    "hyper",
    "tabby",
    "wsl",
    "bash",
    "ghostty",
    "warp",
];

/// Whether `process_name` (e.g. `C:\\...\\WindowsTerminal.exe`, `pwsh.exe`,
/// `iTerm2`) is a known terminal.
pub fn is_terminal(process_name: &str) -> bool {
    let file = process_name
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(process_name)
        .trim()
        .to_ascii_lowercase();
    let stem = file
        .strip_suffix(".exe")
        .or_else(|| file.strip_suffix(".app"))
        .unwrap_or(&file);
    TERMINALS.contains(&stem)
}

/// Characters for which one Left arrow press is known to move exactly one
/// character in ordinary text fields: no surrogate pairs (emoji count as two
/// UTF-16 units in some apps), no combining marks, joiners or complex scripts
/// whose grapheme clusters span several code points.
fn is_simple_char(c: char) -> bool {
    let cp = c as u32;
    match cp {
        0x20..=0x7E => true,
        // Latin-1 .. Cyrillic/Armenian, minus combining diacritics.
        0xA0..=0x058F => !(0x0300..=0x036F).contains(&cp),
        // General punctuation (dashes, smart quotes, ellipsis) minus
        // zero-width / bidi controls and invisible operators.
        0x2010..=0x2027 | 0x2030..=0x205E => true,
        // Currency, letterlike symbols, arrows.
        0x20A0..=0x20BF | 0x2100..=0x214F | 0x2190..=0x21FF => true,
        // CJK punctuation, kana, unified ideographs, Hangul syllables,
        // full-width forms: one code point per visible character.
        0x3000..=0x303F | 0x3040..=0x30FF | 0x4E00..=0x9FFF | 0xAC00..=0xD7A3 => true,
        0xFF01..=0xFFEF => true,
        _ => false,
    }
}

/// Why a swap falls back to copying the other version to the clipboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyReason {
    /// The foreground window changed, or it cannot be determined (non-Windows).
    DifferentWindow,
    /// Keyboard or mouse-button input happened since the paste, or input
    /// tracking is unavailable.
    InputSincePaste,
    TooOld,
    Terminal,
    MultiLine,
    /// Characters whose cursor movement is unpredictable, or too long.
    UnsafeText,
    /// Auto-submit already sent the text, or the paste method did not insert
    /// it as plain text Handy can select back (None / external script).
    PasteMethod,
    /// Modifier keys are still held (Shift+Left would become word selection).
    ModifiersHeld,
}

/// Everything the guard needs to know about the last Handy paste.
#[derive(Clone, Debug)]
pub struct SwapCheck<'a> {
    /// Exactly what was inserted (including any appended trailing space).
    pub inserted_text: &'a str,
    pub age: Duration,
    /// `None` when the foreground window cannot be determined.
    pub same_window: Option<bool>,
    /// `None` when input tracking is unavailable.
    pub input_since_paste: Option<bool>,
    pub process_name: Option<&'a str>,
    /// The paste method types or pastes plain text that Shift+Left can
    /// select back, and auto-submit is off.
    pub paste_method_ok: bool,
    pub modifiers_released: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapPlan {
    /// Select back this many characters, then paste the other version.
    InPlace {
        select_back: usize,
    },
    CopyOnly(CopyReason),
}

pub fn plan_swap(check: &SwapCheck<'_>) -> SwapPlan {
    use CopyReason::*;
    let fail = SwapPlan::CopyOnly;
    if !check.paste_method_ok {
        return fail(PasteMethod);
    }
    if check.age >= MAX_SWAP_AGE {
        return fail(TooOld);
    }
    if check.same_window != Some(true) {
        return fail(DifferentWindow);
    }
    if check.process_name.is_some_and(is_terminal) {
        return fail(Terminal);
    }
    if check.input_since_paste != Some(false) {
        return fail(InputSincePaste);
    }
    if check.inserted_text.contains(['\n', '\r']) {
        return fail(MultiLine);
    }
    let count = check.inserted_text.chars().count();
    if count == 0
        || count > MAX_SELECT_BACK_CHARS
        || !check.inserted_text.chars().all(is_simple_char)
    {
        return fail(UnsafeText);
    }
    if !check.modifiers_released {
        return fail(ModifiersHeld);
    }
    SwapPlan::InPlace { select_back: count }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_check(text: &str) -> SwapCheck<'_> {
        SwapCheck {
            inserted_text: text,
            age: Duration::from_secs(5),
            same_window: Some(true),
            input_since_paste: Some(false),
            process_name: Some(r"C:\Program Files\Slack\slack.exe"),
            paste_method_ok: true,
            modifiers_released: true,
        }
    }

    #[test]
    fn all_guards_pass_selects_back_char_count() {
        assert_eq!(
            plan_swap(&ok_check("Hello, world. ")),
            SwapPlan::InPlace { select_back: 14 }
        );
        // Smart quotes, dashes and accents are one character each.
        assert_eq!(
            plan_swap(&ok_check("Café — “quoted”")),
            SwapPlan::InPlace { select_back: 15 }
        );
    }

    #[test]
    fn each_guard_falls_back_to_copy() {
        use CopyReason::*;
        let cases: Vec<(SwapCheck, CopyReason)> = vec![
            (
                SwapCheck {
                    same_window: Some(false),
                    ..ok_check("hi")
                },
                DifferentWindow,
            ),
            (
                SwapCheck {
                    same_window: None,
                    ..ok_check("hi")
                },
                DifferentWindow,
            ),
            (
                SwapCheck {
                    input_since_paste: Some(true),
                    ..ok_check("hi")
                },
                InputSincePaste,
            ),
            (
                SwapCheck {
                    input_since_paste: None,
                    ..ok_check("hi")
                },
                InputSincePaste,
            ),
            (
                SwapCheck {
                    age: Duration::from_secs(120),
                    ..ok_check("hi")
                },
                TooOld,
            ),
            (
                SwapCheck {
                    process_name: Some("WindowsTerminal.exe"),
                    ..ok_check("hi")
                },
                Terminal,
            ),
            (ok_check("line one\nline two"), MultiLine),
            (ok_check("line one\r\n"), MultiLine),
            (ok_check("party 🎉"), UnsafeText),
            (ok_check("e\u{301}"), UnsafeText),
            (ok_check(""), UnsafeText),
            (
                SwapCheck {
                    paste_method_ok: false,
                    ..ok_check("hi")
                },
                PasteMethod,
            ),
            (
                SwapCheck {
                    modifiers_released: false,
                    ..ok_check("hi")
                },
                ModifiersHeld,
            ),
        ];
        for (check, reason) in cases {
            assert_eq!(plan_swap(&check), SwapPlan::CopyOnly(reason), "{check:?}");
        }
        let long = "a".repeat(MAX_SELECT_BACK_CHARS + 1);
        assert_eq!(plan_swap(&ok_check(&long)), SwapPlan::CopyOnly(UnsafeText));
    }

    #[test]
    fn terminal_detection() {
        for name in [
            "WindowsTerminal.exe",
            r"C:\Windows\System32\conhost.exe",
            "cmd.exe",
            "powershell.exe",
            "PWSH.EXE",
            "wezterm-gui.exe",
            "alacritty.exe",
            "mintty.exe",
            "putty.exe",
            "kitty",
            "iTerm2",
            "Terminal",
            "/Applications/Terminal.app",
        ] {
            assert!(is_terminal(name), "{name} should be a terminal");
        }
        for name in [
            "Code.exe",
            "slack.exe",
            "WINWORD.EXE",
            "chrome.exe",
            "notepad.exe",
            "terminalizer-notes.exe",
        ] {
            assert!(!is_terminal(name), "{name} should not be a terminal");
        }
    }
}
