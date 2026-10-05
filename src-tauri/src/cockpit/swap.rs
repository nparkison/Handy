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
/// ".exe". Terminals embedded in other apps (VS Code, JetBrains IDEs) report
/// the host app's executable and cannot be detected here; nothing else
/// guards against them, which is why a swap must also verify the selection
/// (see `cockpit::swap_last`).
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
    "conemu",
    "conemu64",
    "conemuc",
    "conemuc64",
    "mobaxterm",
    "termius",
    "xshell",
    "securecrt",
    "cmder",
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
/// character in ordinary text fields. An allow-list of scripts that use one
/// code point per caret stop: no surrogate pairs (emoji count as two UTF-16
/// units in some apps), no combining marks (Mn/Me), no format characters (Cf:
/// soft hyphen, ZWJ/ZWNJ, bidi controls), no variation selectors, and no
/// complex scripts whose grapheme clusters span several code points. A wrong
/// count deletes user text, so anything not listed falls back to copying.
fn is_simple_char(c: char) -> bool {
    let cp = c as u32;
    match cp {
        0x20..=0x7E => true,
        // Latin-1 Supplement (minus the soft hyphen, Cf), Latin Extended-A/B,
        // IPA, spacing modifier letters.
        0xA0..=0x2FF => cp != 0xAD,
        // Greek and Coptic (no combining marks in this block).
        0x370..=0x3FF => true,
        // Cyrillic minus its combining marks U+0483..U+0489, and the
        // Cyrillic Supplement.
        0x400..=0x482 | 0x48A..=0x52F => true,
        // General punctuation (dashes, smart quotes, ellipsis) minus
        // zero-width / bidi controls and invisible operators.
        0x2010..=0x2027 | 0x2030..=0x205E => true,
        // Currency, letterlike symbols, arrows.
        0x20A0..=0x20C0 | 0x2100..=0x214F | 0x2190..=0x21FF => true,
        // CJK punctuation minus the combining tone marks U+302A..U+302F.
        0x3000..=0x3029 | 0x3030..=0x303F => true,
        // Hiragana minus the combining voiced marks U+3099/U+309A; Katakana.
        0x3041..=0x3096 | 0x309B..=0x309F | 0x30A0..=0x30FF => true,
        // Unified ideographs and precomposed Hangul syllables.
        0x4E00..=0x9FFF | 0xAC00..=0xD7A3 => true,
        // Full-width forms and signs (half-width forms are left out).
        0xFF01..=0xFF60 | 0xFFE0..=0xFFE6 => true,
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
    // An unknown process could be a terminal: never risk it.
    if check.process_name.is_none_or(is_terminal) {
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
            (
                SwapCheck {
                    process_name: None,
                    ..ok_check("hi")
                },
                Terminal,
            ),
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
    fn invisible_and_combining_characters_are_unsafe() {
        for text in [
            "soft\u{AD}hyphen",
            "\u{0418}\u{0483}", // Cyrillic titlo (Mn)
            "\u{0488}",         // Cyrillic hundred thousands sign (Me)
            "a\u{200D}b",       // ZWJ
            "a\u{200C}b",       // ZWNJ
            "a\u{FE0F}",        // variation selector
            "a\u{202E}b",       // bidi override
            "\u{304B}\u{3099}", // combining kana voiced mark
            "\u{3000}\u{302A}", // ideographic tone mark
            "\u{FF76}\u{FF9E}", // half-width katakana + voiced mark
            "\u{05D0}",         // Hebrew (not allow-listed)
        ] {
            assert_eq!(
                plan_swap(&ok_check(text)),
                SwapPlan::CopyOnly(CopyReason::UnsafeText),
                "{text:?}"
            );
        }
        // Precomposed and single-stop characters are fine.
        for text in [
            "\u{304C}",
            "Привет",
            "Ελλάδα",
            "日本語の文",
            "한국어",
            "naïve",
        ] {
            assert!(
                matches!(plan_swap(&ok_check(text)), SwapPlan::InPlace { .. }),
                "{text:?}"
            );
        }
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
            "ConEmu64.exe",
            "MobaXterm.exe",
            "Termius.exe",
            "Xshell.exe",
            "SecureCRT.exe",
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
