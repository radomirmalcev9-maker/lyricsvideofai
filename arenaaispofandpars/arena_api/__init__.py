"""Unofficial, cookie-authenticated sync/async clients for Arena frontends.

The quickest start is :func:`arena_api.auto_client`: it finds the session of the
browser you are already signed in to, matches its fingerprint, verifies it and
keeps it fresh. This package uses caller-provided credentials — or your own
local browser profile (``auto_client()``, ``session_source="firefox"``); nothing
is collected anywhere else. See the project README for protocol caveats.

On a machine with no profile of yours on it, sign in once by hand --
:func:`arena_api.interactive_login` from Python, or ``arena-api --login`` in a
terminal. It opens a real browser window, waits for you, saves the session to
``state/cookies.json`` and verifies it, after which ``auto_client()`` works with
no arguments at all.

``arena_api.auth`` additionally ships E2E/synthetic-monitoring tooling
(Playwright wrapper, pointer kinematics, determinism init scripts, session
lifecycle management and a challenge *detector*); see
``arena_api.auth.environment`` and ``arena_api.auth.challenge``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .auth import (
        BrowserEnvironment,
        BrowserEnvironmentOptions,
        BrowserSessionManager,
        CaptchaAttempt,
        CaptchaSolver,
        ChallengeHandler,
        ChallengeKind,
        ChallengeObservation,
        ChallengeWaitSolver,
        CompositeSolver,
        DeterminismOptions,
        FirefoxSessionError,
        FirefoxSessionManager,
        FormLoginFlow,
        KinematicMouse,
        KinematicMouseSync,
        LoginResult,
        RecaptchaToolkitSolver,
        SessionManager,
        SessionValidation,
        TrajectoryPlanner,
        alogin_and_save,
        build_init_script,
        collect_firefox_report,
        firefox_profiles_roots,
        format_firefox_report,
        inspect_page,
        inspect_response,
        login_and_save,
    )
    from .auto import (
        AutoOptions,
        AutoSessionManager,
        DirectVerificationToken,
        DiscoveredSession,
        LoginOutcome,
        ProbeResult,
        auto_async_client,
        auto_async_direct_client,
        auto_client,
        auto_direct_client,
        discover_session,
        explain_impersonation,
        impersonation_profile,
        interactive_login,
        profile_for_source,
    )
    from .client import (
        ArenaClient,
        ArenaClient as Client,
        AsyncArenaClient,
        AsyncArenaClient as AsyncClient,
        resolve_impersonation,
    )
    from .direct import (
        AsyncDirectArenaClient,
        AsyncDirectChatSession,
        DirectArenaClient,
        DirectChatSession,
        model_chat_score,
        model_fast_score,
    )
    from .exceptions import (
        ArenaError,
        AuthenticationError,
        CaptchaSolveError,
        ChallengeRequiredError,
        ClientClosedError,
        ModelUnavailableError,
        ProtocolError,
        QueueTimeoutError,
        RateLimitError,
    )
    from .models import (
        BattleResponse,
        BattleStreamChunk,
        GenerationConfig,
        Message,
        ModelInfo,
        QueueEndpoint,
        StreamChunk,
    )
    from .session import AsyncChatSession, ChatSession
    from .warm import (
        describe_daemon,
        stop_daemon,
        warm_enabled,
    )
    from .transport import (
        IMPERSONATE_EXTRA,
        ArenaTransport,
        AsyncArenaTransport,
        SSEParser,
        impersonation_available,
        normalize_cookies,
        parse_cookie_header,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # .auth
    "BrowserEnvironment": (".auth", "BrowserEnvironment"),
    "BrowserEnvironmentOptions": (".auth", "BrowserEnvironmentOptions"),
    "BrowserSessionManager": (".auth", "BrowserSessionManager"),
    "CaptchaAttempt": (".auth", "CaptchaAttempt"),
    "CaptchaSolver": (".auth", "CaptchaSolver"),
    "ChallengeHandler": (".auth", "ChallengeHandler"),
    "ChallengeKind": (".auth", "ChallengeKind"),
    "ChallengeObservation": (".auth", "ChallengeObservation"),
    "ChallengeWaitSolver": (".auth", "ChallengeWaitSolver"),
    "CompositeSolver": (".auth", "CompositeSolver"),
    "DeterminismOptions": (".auth", "DeterminismOptions"),
    "FirefoxSessionError": (".auth", "FirefoxSessionError"),
    "FirefoxSessionManager": (".auth", "FirefoxSessionManager"),
    "FormLoginFlow": (".auth", "FormLoginFlow"),
    "KinematicMouse": (".auth", "KinematicMouse"),
    "KinematicMouseSync": (".auth", "KinematicMouseSync"),
    "LoginResult": (".auth", "LoginResult"),
    "RecaptchaToolkitSolver": (".auth", "RecaptchaToolkitSolver"),
    "SessionManager": (".auth", "SessionManager"),
    "SessionValidation": (".auth", "SessionValidation"),
    "TrajectoryPlanner": (".auth", "TrajectoryPlanner"),
    "alogin_and_save": (".auth", "alogin_and_save"),
    "build_init_script": (".auth", "build_init_script"),
    "collect_firefox_report": (".auth", "collect_firefox_report"),
    "firefox_profiles_roots": (".auth", "firefox_profiles_roots"),
    "format_firefox_report": (".auth", "format_firefox_report"),
    "inspect_page": (".auth", "inspect_page"),
    "inspect_response": (".auth", "inspect_response"),
    "login_and_save": (".auth", "login_and_save"),
    # .client
    "ArenaClient": (".client", "ArenaClient"),
    "AsyncArenaClient": (".client", "AsyncArenaClient"),
    "Client": (".client", "ArenaClient"),
    "AsyncClient": (".client", "AsyncArenaClient"),
    "resolve_impersonation": (".client", "resolve_impersonation"),
    # .direct
    "AsyncDirectArenaClient": (".direct", "AsyncDirectArenaClient"),
    "AsyncDirectChatSession": (".direct", "AsyncDirectChatSession"),
    "DirectArenaClient": (".direct", "DirectArenaClient"),
    "DirectChatSession": (".direct", "DirectChatSession"),
    "model_chat_score": (".direct", "model_chat_score"),
    "model_fast_score": (".direct", "model_fast_score"),
    # .exceptions
    "ArenaError": (".exceptions", "ArenaError"),
    "AuthenticationError": (".exceptions", "AuthenticationError"),
    "CaptchaSolveError": (".exceptions", "CaptchaSolveError"),
    "ChallengeRequiredError": (".exceptions", "ChallengeRequiredError"),
    "ClientClosedError": (".exceptions", "ClientClosedError"),
    "ModelUnavailableError": (".exceptions", "ModelUnavailableError"),
    "ProtocolError": (".exceptions", "ProtocolError"),
    "QueueTimeoutError": (".exceptions", "QueueTimeoutError"),
    "RateLimitError": (".exceptions", "RateLimitError"),
    # .models
    "BattleResponse": (".models", "BattleResponse"),
    "BattleStreamChunk": (".models", "BattleStreamChunk"),
    "GenerationConfig": (".models", "GenerationConfig"),
    "Message": (".models", "Message"),
    "ModelInfo": (".models", "ModelInfo"),
    "QueueEndpoint": (".models", "QueueEndpoint"),
    "StreamChunk": (".models", "StreamChunk"),
    # .session
    "AsyncChatSession": (".session", "AsyncChatSession"),
    "ChatSession": (".session", "ChatSession"),
    # .transport
    "IMPERSONATE_EXTRA": (".transport", "IMPERSONATE_EXTRA"),
    "ArenaTransport": (".transport", "ArenaTransport"),
    "AsyncArenaTransport": (".transport", "AsyncArenaTransport"),
    "SSEParser": (".transport", "SSEParser"),
    "impersonation_available": (".transport", "impersonation_available"),
    "normalize_cookies": (".transport", "normalize_cookies"),
    "parse_cookie_header": (".transport", "parse_cookie_header"),
    # .auto
    "AutoOptions": (".auto", "AutoOptions"),
    "AutoSessionManager": (".auto", "AutoSessionManager"),
    "DirectVerificationToken": (".auto", "DirectVerificationToken"),
    "DiscoveredSession": (".auto", "DiscoveredSession"),
    "LoginOutcome": (".auto", "LoginOutcome"),
    "ProbeResult": (".auto", "ProbeResult"),
    "auto_async_client": (".auto", "auto_async_client"),
    "auto_async_direct_client": (".auto", "auto_async_direct_client"),
    "auto_client": (".auto", "auto_client"),
    "auto_direct_client": (".auto", "auto_direct_client"),
    "discover_session": (".auto", "discover_session"),
    "explain_impersonation": (".auto", "explain_impersonation"),
    "impersonation_profile": (".auto", "impersonation_profile"),
    "interactive_login": (".auto", "interactive_login"),
    "profile_for_source": (".auto", "profile_for_source"),
    # .warm
    "describe_daemon": (".warm", "describe_daemon"),
    "stop_daemon": (".warm", "stop_daemon"),
    "warm_enabled": (".warm", "warm_enabled"),
}

_SUBMODULES: frozenset[str] = frozenset(
    {
        "auth",
        "auto",
        "client",
        "direct",
        "exceptions",
        "models",
        "session",
        "transport",
        "warm",
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
    "ArenaClient",
    "ArenaError",
    "ArenaTransport",
    "AsyncArenaClient",
    "AsyncArenaTransport",
    "AsyncChatSession",
    "AsyncClient",
    "AsyncDirectArenaClient",
    "AsyncDirectChatSession",
    "AuthenticationError",
    "AutoOptions",
    "AutoSessionManager",
    "BattleResponse",
    "BattleStreamChunk",
    "BrowserEnvironment",
    "BrowserEnvironmentOptions",
    "BrowserSessionManager",
    "CaptchaAttempt",
    "CaptchaSolveError",
    "CaptchaSolver",
    "ChallengeHandler",
    "ChallengeKind",
    "ChallengeObservation",
    "ChallengeWaitSolver",
    "CompositeSolver",
    "ChallengeRequiredError",
    "ChatSession",
    "IMPERSONATE_EXTRA",
    "Client",
    "ClientClosedError",
    "DeterminismOptions",
    "DirectArenaClient",
    "DirectChatSession",
    "DirectVerificationToken",
    "DiscoveredSession",
    "LoginOutcome",
    "FirefoxSessionError",
    "FirefoxSessionManager",
    "collect_firefox_report",
    "firefox_profiles_roots",
    "format_firefox_report",
    "FormLoginFlow",
    "GenerationConfig",
    "KinematicMouse",
    "explain_impersonation",
    "impersonation_available",
    "impersonation_profile",
    "KinematicMouseSync",
    "LoginResult",
    "Message",
    "ModelInfo",
    "ModelUnavailableError",
    "ProbeResult",
    "ProtocolError",
    "profile_for_source",
    "resolve_impersonation",
    "QueueEndpoint",
    "QueueTimeoutError",
    "RateLimitError",
    "RecaptchaToolkitSolver",
    "SSEParser",
    "SessionManager",
    "SessionValidation",
    "StreamChunk",
    "TrajectoryPlanner",
    "auto_async_client",
    "auto_async_direct_client",
    "auto_client",
    "auto_direct_client",
    "alogin_and_save",
    "build_init_script",
    "describe_daemon",
    "discover_session",
    "interactive_login",
    "inspect_page",
    "inspect_response",
    "login_and_save",
    "model_chat_score",
    "model_fast_score",
    "normalize_cookies",
    "parse_cookie_header",
    "stop_daemon",
    "warm_enabled",
]
