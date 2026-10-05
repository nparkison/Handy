//! Cleanup deadline: race the LLM cleanup against a time limit that starts
//! when the request is sent (not at key release, so a slow local STT model
//! does not eat the budget).
//!
//! The cleanup runs as its own task. On a miss the caller pastes the original
//! text right away and keeps the task handle: the request continues in the
//! background (bounded by [`BACKGROUND_CAP`]) and only ever writes to History.

use futures_util::future::{select, Either};
use std::future::Future;
use std::sync::Mutex;
use std::time::Duration;
use tokio::sync::oneshot;

/// After a miss, the background request is abandoned after this long.
pub const BACKGROUND_CAP: Duration = Duration::from_secs(30);

/// A late result arriving later than this after the miss updates History
/// silently, without a "Cleanup ready" notice.
pub const LATE_NOTICE_WINDOW: Duration = Duration::from_secs(15);

/// How one cleanup attempt ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CleanupOutcome {
    /// Nothing to do: blank text, or no provider/model/prompt configured.
    Skipped,
    /// A request was attempted and produced nothing usable.
    Failed,
    Cleaned(String),
}

impl CleanupOutcome {
    pub fn cleaned(self) -> Option<String> {
        match self {
            CleanupOutcome::Cleaned(text) => Some(text),
            _ => None,
        }
    }
}

/// Fired by the cleanup task right before its first LLM request goes out.
/// Later calls are no-ops. Dropping it unfired means "no request was made".
pub struct RequestSent(Mutex<Option<oneshot::Sender<()>>>);

impl RequestSent {
    pub fn new() -> (Self, oneshot::Receiver<()>) {
        let (tx, rx) = oneshot::channel();
        (Self(Mutex::new(Some(tx))), rx)
    }

    /// A signal nobody listens to, for callers without a deadline.
    pub fn detached() -> Self {
        Self(Mutex::new(None))
    }

    pub fn mark(&self) {
        let sender = self.0.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(sender) = sender {
            let _ = sender.send(());
        }
    }
}

/// Owns a spawned task and aborts it when dropped (cancel, background cap),
/// so an abandoned cleanup request does not linger until the HTTP timeout.
pub struct AbortOnDrop<T>(pub tauri::async_runtime::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Wait for `task` under the cleanup deadline.
///
/// Returns `Some(output)` when the task finished in time (or never sent a
/// request, or there is no deadline), and `None` on a miss: the deadline
/// elapsed after the request was sent. On a miss the task is left running and
/// the caller still owns it.
pub async fn race_with_deadline<F>(
    task: &mut F,
    sent: oneshot::Receiver<()>,
    deadline: Option<Duration>,
) -> Option<F::Output>
where
    F: Future + Unpin,
{
    let Some(deadline) = deadline else {
        return Some(task.await);
    };

    // Phase 1: until the request goes out there is no clock running.
    match select(&mut *task, sent).await {
        Either::Left((output, _)) => return Some(output),
        // Sent, or the signal was dropped unfired (the task is finishing
        // without a request): either way only the task is left to wait for.
        Either::Right((Ok(()), _)) => {}
        Either::Right((Err(_), _)) => return Some(task.await),
    }

    // Phase 2: the request is in flight; the clock runs.
    tokio::time::timeout(deadline, &mut *task).await.ok()
}

/// Whether a late cleanup result should offer "Cleanup ready · Swap".
/// Spec: only within [`LATE_NOTICE_WINDOW`] of the miss, only when no other
/// dictation has started since and Handy is idle (dictation always wins), and
/// only when the user has a way to swap.
pub fn should_offer_late_swap(
    since_miss: Duration,
    dictation_started_since: bool,
    pipeline_busy: bool,
    swap_reachable: bool,
) -> bool {
    since_miss <= LATE_NOTICE_WINDOW && !dictation_started_since && !pipeline_busy && swap_reachable
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future;
    use std::pin::Pin;

    fn run<T>(f: impl Future<Output = T>) -> T {
        tauri::async_runtime::block_on(f)
    }

    type Boxed<T> = Pin<Box<dyn Future<Output = T> + Send>>;

    fn after(ms: u64, value: &'static str) -> Boxed<&'static str> {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(ms)).await;
            value
        })
    }

    #[test]
    fn finishes_before_request_is_sent() {
        let (_signal, rx) = RequestSent::new();
        let mut task: Boxed<&str> = Box::pin(future::ready("skipped"));
        let out = run(race_with_deadline(
            &mut task,
            rx,
            Some(Duration::from_millis(10)),
        ));
        assert_eq!(out, Some("skipped"));
    }

    #[test]
    fn fast_response_after_send_wins() {
        run(async {
            let (signal, rx) = RequestSent::new();
            signal.mark();
            let mut task = after(5, "cleaned");
            let out = race_with_deadline(&mut task, rx, Some(Duration::from_millis(500))).await;
            assert_eq!(out, Some("cleaned"));
        });
    }

    #[test]
    fn slow_response_misses_and_task_keeps_running() {
        run(async {
            let (signal, rx) = RequestSent::new();
            signal.mark();
            let mut task = after(200, "late");
            let out = race_with_deadline(&mut task, rx, Some(Duration::from_millis(20))).await;
            assert_eq!(out, None);
            // The caller still owns the task and can finish it in the background.
            assert_eq!(task.await, "late");
        });
    }

    #[test]
    fn clock_starts_at_send_not_at_spawn() {
        run(async {
            let (signal, rx) = RequestSent::new();
            // 60 ms of pre-request work (e.g. waiting for the screenshot),
            // then a 10 ms request: well over the 40 ms limit in total, but
            // the request itself is fast, so it must not count as a miss.
            let mut task: Boxed<&str> = Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(60)).await;
                signal.mark();
                tokio::time::sleep(Duration::from_millis(10)).await;
                "cleaned"
            });
            let out = race_with_deadline(&mut task, rx, Some(Duration::from_millis(40))).await;
            assert_eq!(out, Some("cleaned"));
        });
    }

    #[test]
    fn dropped_signal_waits_for_task_without_deadline() {
        run(async {
            let (signal, rx) = RequestSent::new();
            drop(signal);
            let mut task = after(30, "no request");
            let out = race_with_deadline(&mut task, rx, Some(Duration::from_millis(5))).await;
            assert_eq!(out, Some("no request"));
        });
    }

    #[test]
    fn no_limit_always_waits() {
        run(async {
            let (signal, rx) = RequestSent::new();
            signal.mark();
            let mut task = after(30, "slow");
            assert_eq!(race_with_deadline(&mut task, rx, None).await, Some("slow"));
        });
    }

    #[test]
    fn mark_is_idempotent() {
        let (signal, _rx) = RequestSent::new();
        signal.mark();
        signal.mark();
        RequestSent::detached().mark();
    }

    #[test]
    fn late_swap_offer_rules() {
        let quick = Duration::from_secs(3);
        assert!(should_offer_late_swap(quick, false, false, true));
        assert!(!should_offer_late_swap(
            Duration::from_secs(16),
            false,
            false,
            true
        ));
        assert!(!should_offer_late_swap(quick, true, false, true));
        assert!(!should_offer_late_swap(quick, false, true, true));
        assert!(!should_offer_late_swap(quick, false, false, false));
    }
}
