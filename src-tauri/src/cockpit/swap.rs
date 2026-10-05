//! Swap-last safety guards (pure logic).
//!
//! An in-place swap deletes the text Handy just pasted by selecting it back
//! with Shift+Left × N and pasting the other version over the selection. It is
//! never done with Ctrl+Z (Word/Docs/Slack merge undo groups, and Ctrl+Z
//! suspends jobs in terminals) or with backspace runs. A wrong delete destroys
//! user text, so every guard must pass; otherwise the other version is only
//! copied to the clipboard.
//!
//! ## Residual risk
//! The selection is verified (copied with Ctrl+Insert and compared with what
//! Handy inserted) before anything is pasted, so text is only replaced when
//! the selection is exactly Handy's paste. The keys used to select and copy
//! (Shift+Left × N, Ctrl+Insert, and Right to collapse a wrong selection) are
//! sent before that verification, though. Known terminals, IDEs with embedded
//! terminals, spreadsheets and modal editors are excluded by executable name,
//! but an unlisted app whose focused element is not a text field (an embedded
//! terminal or grid in some other app) still receives those keys: they can
//! move its cursor or selection, or reach a shell as escape sequences.

use std::time::Duration;

/// In-place swaps are only attempted this soon after the paste.
pub const MAX_SWAP_AGE: Duration = Duration::from_secs(120);

/// Selecting back more than this many characters is slow and fragile.
pub const MAX_SELECT_BACK_CHARS: usize = 500;

/// Executables (Windows) and app names (macOS/Linux) that are terminals.
/// Matched case-insensitively against the process file name, with or without
/// ".exe". Terminals embedded in other apps report the host app's executable;
/// the known hosts are in [`NON_TEXT_HOSTS`].
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

/// Apps whose focused element is often not a plain text field: IDEs with an
/// embedded terminal (their terminal panel reports the IDE's executable),
/// spreadsheets (Shift+Left extends a cell range in Ready mode) and modal
/// editors (Shift+Left is a motion in normal mode). Swap last only copies in
/// them. Plain editors without an embedded terminal (Notepad, Notepad++,
/// Sublime Text) stay allowed.
const NON_TEXT_HOSTS: &[&str] = &[
    // VS Code and its forks.
    "code",
    "code - insiders",
    "codium",
    "vscodium",
    "cursor",
    "windsurf",
    "trae",
    "zed",
    // JetBrains IDEs (32- and 64-bit launchers) and Android Studio.
    "idea",
    "idea64",
    "pycharm",
    "pycharm64",
    "webstorm",
    "webstorm64",
    "rider",
    "rider64",
    "clion",
    "clion64",
    "goland",
    "goland64",
    "phpstorm",
    "phpstorm64",
    "rubymine",
    "rubymine64",
    "datagrip",
    "datagrip64",
    "rustrover",
    "rustrover64",
    "dataspell",
    "dataspell64",
    "studio",
    "studio64",
    "fleet",
    // Visual Studio.
    "devenv",
    // Spreadsheets.
    "excel",
    "et",
    // Modal / terminal-hosting editors.
    "gvim",
    "nvim-qt",
    "neovide",
    "emacs",
    "runemacs",
];

/// Lower-cased file name of `process_name` without ".exe" / ".app".
fn process_stem(process_name: &str) -> String {
    let file = process_name
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(process_name)
        .trim()
        .to_ascii_lowercase();
    file.strip_suffix(".exe")
        .or_else(|| file.strip_suffix(".app"))
        .map(str::to_string)
        .unwrap_or(file)
}

/// Whether `process_name` (e.g. `C:\\...\\WindowsTerminal.exe`, `pwsh.exe`,
/// `iTerm2`) is a known terminal.
pub fn is_terminal(process_name: &str) -> bool {
    TERMINALS.contains(&process_stem(process_name).as_str())
}

/// Whether `process_name` is a known app whose focus may not be a text field
/// (see [`NON_TEXT_HOSTS`]).
pub fn is_non_text_host(process_name: &str) -> bool {
    NON_TEXT_HOSTS.contains(&process_stem(process_name).as_str())
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
    /// An IDE with an embedded terminal, a spreadsheet or a modal editor.
    NonTextHost,
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
    pub process_name: Option<&'a str>,
    /// The paste method types or pastes plain text that Shift+Left can
    /// select back, and auto-submit is off.
    pub paste_method_ok: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapPlan {
    /// Select back this many characters, then paste the other version.
    InPlace {
        select_back: usize,
    },
    CopyOnly(CopyReason),
}

/// Decide how to swap. The cheap guards on `check` run first; the costly
/// probes only run once they all passed, in this order:
/// `modifiers_released` (waits for the user to let go of Shift/Ctrl/Alt/Win),
/// then `input_since_paste` (`None` when input tracking is unavailable; may
/// send a liveness probe and wait for it).
pub fn plan_swap(
    check: &SwapCheck<'_>,
    modifiers_released: impl FnOnce() -> bool,
    input_since_paste: impl FnOnce() -> Option<bool>,
) -> SwapPlan {
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
    let Some(process) = check.process_name else {
        return fail(Terminal);
    };
    if is_terminal(process) {
        return fail(Terminal);
    }
    if is_non_text_host(process) {
        return fail(NonTextHost);
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
    if !modifiers_released() {
        return fail(ModifiersHeld);
    }
    if input_since_paste() != Some(false) {
        return fail(InputSincePaste);
    }
    SwapPlan::InPlace { select_back: count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn ok_check(text: &str) -> SwapCheck<'_> {
        SwapCheck {
            inserted_text: text,
            age: Duration::from_secs(5),
            same_window: Some(true),
            process_name: Some(r"C:\Program Files\Slack\slack.exe"),
            paste_method_ok: true,
        }
    }

    /// Plan with the probes reporting "released" and "no input".
    fn plan(check: &SwapCheck<'_>) -> SwapPlan {
        plan_swap(check, || true, || Some(false))
    }

    #[test]
    fn all_guards_pass_selects_back_char_count() {
        assert_eq!(
            plan(&ok_check("Hello, world. ")),
            SwapPlan::InPlace { select_back: 14 }
        );
        // Smart quotes, dashes and accents are one character each.
        assert_eq!(
            plan(&ok_check("Café — “quoted”")),
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
            (
                SwapCheck {
                    process_name: Some(
                        r"C:\Users\me\AppData\Local\Programs\Microsoft VS Code\Code.exe",
                    ),
                    ..ok_check("hi")
                },
                NonTextHost,
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
        ];
        for (check, reason) in cases {
            assert_eq!(plan(&check), SwapPlan::CopyOnly(reason), "{check:?}");
        }
        let long = "a".repeat(MAX_SELECT_BACK_CHARS + 1);
        assert_eq!(plan(&ok_check(&long)), SwapPlan::CopyOnly(UnsafeText));

        let check = ok_check("hi");
        assert_eq!(
            plan_swap(&check, || false, || Some(false)),
            SwapPlan::CopyOnly(ModifiersHeld)
        );
        assert_eq!(
            plan_swap(&check, || true, || Some(true)),
            SwapPlan::CopyOnly(InputSincePaste)
        );
        // Input tracking unavailable.
        assert_eq!(
            plan_swap(&check, || true, || None),
            SwapPlan::CopyOnly(InputSincePaste)
        );
    }

    #[test]
    fn costly_probes_only_run_after_every_cheap_guard_passed() {
        let modifiers = Cell::new(0);
        let input = Cell::new(0);
        let probe = |check: &SwapCheck<'_>, released: bool| {
            plan_swap(
                check,
                || {
                    modifiers.set(modifiers.get() + 1);
                    released
                },
                || {
                    input.set(input.get() + 1);
                    Some(false)
                },
            )
        };
        // A cheap guard fails: neither probe runs.
        for check in [
            SwapCheck {
                paste_method_ok: false,
                ..ok_check("hi")
            },
            SwapCheck {
                same_window: Some(false),
                ..ok_check("hi")
            },
            SwapCheck {
                process_name: Some("EXCEL.EXE"),
                ..ok_check("hi")
            },
            ok_check("two\nlines"),
        ] {
            assert!(matches!(probe(&check, true), SwapPlan::CopyOnly(_)));
        }
        assert_eq!((modifiers.get(), input.get()), (0, 0));
        // Modifiers still held: the input probe does not run.
        assert_eq!(
            probe(&ok_check("hi"), false),
            SwapPlan::CopyOnly(CopyReason::ModifiersHeld)
        );
        assert_eq!((modifiers.get(), input.get()), (1, 0));
        // Everything passes: both run, modifiers first.
        assert!(matches!(
            probe(&ok_check("hi"), true),
            SwapPlan::InPlace { .. }
        ));
        assert_eq!((modifiers.get(), input.get()), (2, 1));
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
                plan(&ok_check(text)),
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
                matches!(plan(&ok_check(text)), SwapPlan::InPlace { .. }),
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

    #[test]
    fn ide_spreadsheet_and_modal_editor_detection() {
        for name in [
            "Code.exe",
            r"C:\Users\me\AppData\Local\Programs\Microsoft VS Code Insiders\Code - Insiders.exe",
            "Cursor.exe",
            "Windsurf.exe",
            r"C:\Program Files\JetBrains\IntelliJ IDEA 2025.2\bin\idea64.exe",
            "pycharm64.exe",
            "webstorm64.exe",
            "rider64.exe",
            "clion64.exe",
            "goland64.exe",
            "studio64.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\devenv.exe",
            r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
            "gvim.exe",
        ] {
            assert!(is_non_text_host(name), "{name} should be copy-only");
            assert_eq!(
                plan(&SwapCheck {
                    process_name: Some(name),
                    ..ok_check("hi")
                }),
                SwapPlan::CopyOnly(CopyReason::NonTextHost),
                "{name}"
            );
        }
        for name in [
            "sublime_text.exe",
            "notepad.exe",
            "notepad++.exe",
            "slack.exe",
            "WINWORD.EXE",
            "chrome.exe",
            "Obsidian.exe",
            "codex-notes.exe",
        ] {
            assert!(!is_non_text_host(name), "{name} should stay allowed");
        }
    }
}
