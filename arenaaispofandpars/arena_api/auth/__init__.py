"""Session extraction and browser-session helpers for the Arena clients.

Two backends live here:

* :mod:`arena_api.auth.firefox_session` reuses the signed-in session of your
  local Firefox profile (stdlib-only, ``rookiepy`` optional). Every profile of
  every root is ranked -- snap, Flatpak, portable installs included -- and the
  one that actually stores cookies for the target domains wins, because the
  profile ``profiles.ini`` calls the default often is not that one.
* :mod:`arena_api.auth.diagnostics` answers "why was no session found?" with a
  report that is safe to paste: hosts, names, paths, flags, expiry, value
  lengths and value fingerprints, never a cookie value.
* :mod:`arena_api.auth.session` drives a real Playwright sign-in with your own
  credentials and persists the result as ``cookies.json`` /
  ``storage_state``, with :mod:`arena_api.auth.browser`,
  :mod:`arena_api.auth.environment` and :mod:`arena_api.auth.kinematics`
  providing the Playwright wrapper, determinism init scripts and pointer
  kinematics respectively.

:mod:`arena_api.auth.challenge` detects browser verification challenges,
captures diagnostics, and turns them into an actionable
:class:`~arena_api.exceptions.ChallengeRequiredError` with remediation
guidance -- which is what a synthetic monitor actually needs.

:mod:`arena_api.auth.login` covers the one step nobody can automate away: it
opens a headed browser, lets a person sign in, and persists the session as a
Playwright ``storage_state`` file that later runs discover on their own
(``login_and_save()``, or ``arena-api --login`` from the bootstrap layer).

:mod:`arena_api.auth.captcha` is the opt-in bridge to
:mod:`arena_api._vendor.recaptcha_toolkit`: hand a ``RecaptchaToolkitSolver`` to
a :class:`~arena_api.auth.challenge.ChallengeHandler`, a
:class:`~arena_api.auth.browser.BrowserEnvironmentOptions` or a
:class:`~arena_api.auth.session.BrowserSessionManager` and a detected reCAPTCHA
is solved instead of only reported. The kinds that have no token to produce are
covered too: ``ChallengeWaitSolver`` waits Turnstile and a managed Cloudflare
interstitial out, and hands hCaptcha/DataDome to a person in a headed window;
``CompositeSolver`` dispatches between the two. Nothing is active until you ask.

Heavy dependencies (``playwright``, ``curl_cffi``, ``numpy``,
``arena_api._vendor.recaptcha_toolkit``) are imported lazily inside the
functions that need them, so importing this package stays cheap and works
without them installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .autosolve import (
        AutoSolveOutcome,
        BrowserChallengeAutoSolver,
        auto_solve_challenge_on_page,
        decode_mp3_via_browser_webaudio,
        encode_pcm16le_to_flac,
        parse_google_speech_response,
        transcribe_pcm16_google,
    )
    from .browser import (
        SYNTHETIC_MONITOR_HEADER,
        BrowserEnvironment,
        BrowserEnvironmentOptions,
        run_sync,
    )
    from .captcha import (
        TOOLKIT_PACKAGE,
        WAITABLE_CHALLENGE_KINDS,
        CaptchaAttempt,
        CaptchaSolver,
        ChallengeWaitSolver,
        CompositeSolver,
        RecaptchaToolkitSolver,
        describe_solver,
        detect_recaptcha_version,
        detect_recaptcha_version_sync,
        observation_targets_recaptcha,
        solver_for,
    )
    from .challenge import (
        CLOUDFLARE_FRAME_HOSTS,
        DATADOME_FRAME_HOSTS,
        DATADOME_HEADERS,
        DATADOME_SELECTORS,
        HCAPTCHA_SELECTORS,
        RECAPTCHA_SELECTORS,
        REMEDIATION_TEMPLATE,
        SOLVABLE_CHALLENGE_KINDS,
        STATUS_TEXT_MARKERS,
        TURNSTILE_SELECTORS,
        ChallengeHandler,
        ChallengeKind,
        ChallengeObservation,
        ChallengeSignal,
        inspect_page,
        inspect_page_sync,
        inspect_response,
    )
    from .diagnostics import (
        CookieFingerprint,
        collect_firefox_report,
        format_firefox_report,
        report_as_json,
    )
    from .environment import (
        ANIMATION_FREEZE_CSS,
        DEFAULT_INIT_SCRIPT,
        HIDE_WEBDRIVER_SCRIPT,
        DeterminismOptions,
        build_init_script,
    )
    from .firefox_session import (
        DEFAULT_COOKIE_DOMAINS,
        FIREFOX_PROFILE_ENV,
        FIREFOX_ROOT_ENV,
        FirefoxSessionError,
        FirefoxSessionManager,
        SessionManager,
        firefox_profiles_roots,
    )
    from .kinematics import (
        KinematicMouse,
        KinematicMouseSync,
        MouseSample,
        Point,
        Trajectory,
        TrajectoryPlanner,
        cubic_bezier_point,
        fitts_duration,
        resolve_element_center,
    )
    from .login import (
        DEFAULT_STATE_PATH,
        STATE_FILE_MODE,
        LoginResult,
        alogin_and_save,
        login_and_save,
        login_environment_options,
    )
    from .session import (
        BrowserSessionManager,
        CurlCffiProbe,
        FormLoginFlow,
        LoginFlow,
        RawResponse,
        SessionProbe,
        SessionValidation,
        chromium_user_agent,
        normalize_state_payload,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # .autosolve
    "AutoSolveOutcome": (".autosolve", "AutoSolveOutcome"),
    "BrowserChallengeAutoSolver": (".autosolve", "BrowserChallengeAutoSolver"),
    "auto_solve_challenge_on_page": (".autosolve", "auto_solve_challenge_on_page"),
    "decode_mp3_via_browser_webaudio": (".autosolve", "decode_mp3_via_browser_webaudio"),
    "encode_pcm16le_to_flac": (".autosolve", "encode_pcm16le_to_flac"),
    "parse_google_speech_response": (".autosolve", "parse_google_speech_response"),
    "transcribe_pcm16_google": (".autosolve", "transcribe_pcm16_google"),
    # .browser
    "SYNTHETIC_MONITOR_HEADER": (".browser", "SYNTHETIC_MONITOR_HEADER"),
    "BrowserEnvironment": (".browser", "BrowserEnvironment"),
    "BrowserEnvironmentOptions": (".browser", "BrowserEnvironmentOptions"),
    "run_sync": (".browser", "run_sync"),
    # .captcha
    "TOOLKIT_PACKAGE": (".captcha", "TOOLKIT_PACKAGE"),
    "WAITABLE_CHALLENGE_KINDS": (".captcha", "WAITABLE_CHALLENGE_KINDS"),
    "CaptchaAttempt": (".captcha", "CaptchaAttempt"),
    "CaptchaSolver": (".captcha", "CaptchaSolver"),
    "ChallengeWaitSolver": (".captcha", "ChallengeWaitSolver"),
    "CompositeSolver": (".captcha", "CompositeSolver"),
    "RecaptchaToolkitSolver": (".captcha", "RecaptchaToolkitSolver"),
    "describe_solver": (".captcha", "describe_solver"),
    "detect_recaptcha_version": (".captcha", "detect_recaptcha_version"),
    "detect_recaptcha_version_sync": (".captcha", "detect_recaptcha_version_sync"),
    "observation_targets_recaptcha": (".captcha", "observation_targets_recaptcha"),
    "solver_for": (".captcha", "solver_for"),
    # .challenge
    "CLOUDFLARE_FRAME_HOSTS": (".challenge", "CLOUDFLARE_FRAME_HOSTS"),
    "DATADOME_FRAME_HOSTS": (".challenge", "DATADOME_FRAME_HOSTS"),
    "DATADOME_HEADERS": (".challenge", "DATADOME_HEADERS"),
    "DATADOME_SELECTORS": (".challenge", "DATADOME_SELECTORS"),
    "HCAPTCHA_SELECTORS": (".challenge", "HCAPTCHA_SELECTORS"),
    "RECAPTCHA_SELECTORS": (".challenge", "RECAPTCHA_SELECTORS"),
    "REMEDIATION_TEMPLATE": (".challenge", "REMEDIATION_TEMPLATE"),
    "SOLVABLE_CHALLENGE_KINDS": (".challenge", "SOLVABLE_CHALLENGE_KINDS"),
    "STATUS_TEXT_MARKERS": (".challenge", "STATUS_TEXT_MARKERS"),
    "TURNSTILE_SELECTORS": (".challenge", "TURNSTILE_SELECTORS"),
    "ChallengeHandler": (".challenge", "ChallengeHandler"),
    "ChallengeKind": (".challenge", "ChallengeKind"),
    "ChallengeObservation": (".challenge", "ChallengeObservation"),
    "ChallengeSignal": (".challenge", "ChallengeSignal"),
    "inspect_page": (".challenge", "inspect_page"),
    "inspect_page_sync": (".challenge", "inspect_page_sync"),
    "inspect_response": (".challenge", "inspect_response"),
    # .environment
    "ANIMATION_FREEZE_CSS": (".environment", "ANIMATION_FREEZE_CSS"),
    "DEFAULT_INIT_SCRIPT": (".environment", "DEFAULT_INIT_SCRIPT"),
    "HIDE_WEBDRIVER_SCRIPT": (".environment", "HIDE_WEBDRIVER_SCRIPT"),
    "DeterminismOptions": (".environment", "DeterminismOptions"),
    "build_init_script": (".environment", "build_init_script"),
    # .diagnostics
    "CookieFingerprint": (".diagnostics", "CookieFingerprint"),
    "collect_firefox_report": (".diagnostics", "collect_firefox_report"),
    "format_firefox_report": (".diagnostics", "format_firefox_report"),
    "report_as_json": (".diagnostics", "report_as_json"),
    # .firefox_session
    "DEFAULT_COOKIE_DOMAINS": (".firefox_session", "DEFAULT_COOKIE_DOMAINS"),
    "FIREFOX_PROFILE_ENV": (".firefox_session", "FIREFOX_PROFILE_ENV"),
    "FIREFOX_ROOT_ENV": (".firefox_session", "FIREFOX_ROOT_ENV"),
    "FirefoxSessionError": (".firefox_session", "FirefoxSessionError"),
    "FirefoxSessionManager": (".firefox_session", "FirefoxSessionManager"),
    "SessionManager": (".firefox_session", "SessionManager"),
    "firefox_profiles_roots": (".firefox_session", "firefox_profiles_roots"),
    # .kinematics
    "KinematicMouse": (".kinematics", "KinematicMouse"),
    "KinematicMouseSync": (".kinematics", "KinematicMouseSync"),
    "MouseSample": (".kinematics", "MouseSample"),
    "Point": (".kinematics", "Point"),
    "Trajectory": (".kinematics", "Trajectory"),
    "TrajectoryPlanner": (".kinematics", "TrajectoryPlanner"),
    "cubic_bezier_point": (".kinematics", "cubic_bezier_point"),
    "fitts_duration": (".kinematics", "fitts_duration"),
    "resolve_element_center": (".kinematics", "resolve_element_center"),
    # .login
    "DEFAULT_STATE_PATH": (".login", "DEFAULT_STATE_PATH"),
    "STATE_FILE_MODE": (".login", "STATE_FILE_MODE"),
    "LoginResult": (".login", "LoginResult"),
    "alogin_and_save": (".login", "alogin_and_save"),
    "login_and_save": (".login", "login_and_save"),
    "login_environment_options": (".login", "login_environment_options"),
    # .session
    "BrowserSessionManager": (".session", "BrowserSessionManager"),
    "CurlCffiProbe": (".session", "CurlCffiProbe"),
    "FormLoginFlow": (".session", "FormLoginFlow"),
    "LoginFlow": (".session", "LoginFlow"),
    "RawResponse": (".session", "RawResponse"),
    "SessionProbe": (".session", "SessionProbe"),
    "SessionValidation": (".session", "SessionValidation"),
    "chromium_user_agent": (".session", "chromium_user_agent"),
    "normalize_state_payload": (".session", "normalize_state_payload"),
}

_SUBMODULES: frozenset[str] = frozenset(
    {
        "autosolve",
        "browser",
        "captcha",
        "challenge",
        "diagnostics",
        "environment",
        "firefox_session",
        "kinematics",
        "login",
        "proxy",
        "session",
    }
)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        mod_name, attr_name = target
        module = importlib.import_module(mod_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ANIMATION_FREEZE_CSS",
    "AutoSolveOutcome",
    "BrowserChallengeAutoSolver",
    "BrowserEnvironment",
    "BrowserEnvironmentOptions",
    "BrowserSessionManager",
    "CLOUDFLARE_FRAME_HOSTS",
    "DATADOME_FRAME_HOSTS",
    "DATADOME_HEADERS",
    "DATADOME_SELECTORS",
    "CaptchaAttempt",
    "CaptchaSolver",
    "ChallengeWaitSolver",
    "CompositeSolver",
    "ChallengeHandler",
    "ChallengeKind",
    "ChallengeObservation",
    "ChallengeSignal",
    "CurlCffiProbe",
    "DEFAULT_COOKIE_DOMAINS",
    "DEFAULT_INIT_SCRIPT",
    "DEFAULT_STATE_PATH",
    "DeterminismOptions",
    "FirefoxSessionError",
    "CookieFingerprint",
    "FIREFOX_PROFILE_ENV",
    "FIREFOX_ROOT_ENV",
    "FirefoxSessionManager",
    "collect_firefox_report",
    "firefox_profiles_roots",
    "format_firefox_report",
    "report_as_json",
    "FormLoginFlow",
    "HCAPTCHA_SELECTORS",
    "HIDE_WEBDRIVER_SCRIPT",
    "KinematicMouse",
    "KinematicMouseSync",
    "LoginFlow",
    "LoginResult",
    "MouseSample",
    "Point",
    "RECAPTCHA_SELECTORS",
    "REMEDIATION_TEMPLATE",
    "RawResponse",
    "RecaptchaToolkitSolver",
    "SOLVABLE_CHALLENGE_KINDS",
    "STATUS_TEXT_MARKERS",
    "STATE_FILE_MODE",
    "TOOLKIT_PACKAGE",
    "WAITABLE_CHALLENGE_KINDS",
    "SYNTHETIC_MONITOR_HEADER",
    "SessionManager",
    "SessionProbe",
    "SessionValidation",
    "TURNSTILE_SELECTORS",
    "Trajectory",
    "TrajectoryPlanner",
    "build_init_script",
    "alogin_and_save",
    "auto_solve_challenge_on_page",
    "chromium_user_agent",
    "cubic_bezier_point",
    "decode_mp3_via_browser_webaudio",
    "describe_solver",
    "detect_recaptcha_version",
    "detect_recaptcha_version_sync",
    "encode_pcm16le_to_flac",
    "fitts_duration",
    "inspect_page",
    "inspect_page_sync",
    "login_and_save",
    "login_environment_options",
    "inspect_response",
    "normalize_state_payload",
    "parse_google_speech_response",
    "solver_for",
    "observation_targets_recaptcha",
    "resolve_element_center",
    "run_sync",
    "transcribe_pcm16_google",
]
