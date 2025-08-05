"""Error handling and recovery implementations."""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Protocol


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class RecoveryStrategy(Enum):
    """Recovery strategy options."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error: Exception
    severity: ErrorSeverity
    timestamp: float
    function_name: str
    args: tuple
    kwargs: dict
    retry_count: int = 0


class ErrorHandler(Protocol):
    """Protocol for error handling strategies."""

    def handle_error(self, context: ErrorContext) -> Any:
        """Handle an error with the given context."""
        ...

    def can_recover(self, context: ErrorContext) -> bool:
        """Check if recovery is possible for this error."""
        ...


class AutomaticErrorRecovery:
    """Automatic error recovery implementation."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_history: List[ErrorContext] = []
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0
        }

    def can_recover(self, context: ErrorContext) -> bool:
        """Check if automatic recovery is possible."""
        # Don't retry for certain critical errors
        if isinstance(context.error, (MemoryError, SystemExit, KeyboardInterrupt)):
            return False

        # Check retry limit
        return context.retry_count < self.max_retries

    def handle_error(self, context: ErrorContext) -> Any:
        """Handle error with automatic recovery."""
        self.error_history.append(context)
        self.recovery_stats['total_errors'] += 1

        if not self.can_recover(context):
            self.recovery_stats['failed_recoveries'] += 1
            raise context.error

        # Calculate backoff delay
        delay = self.backoff_factor ** context.retry_count
        logging.warning(f"Retrying {context.function_name} after {delay}s delay")

        time.sleep(delay)

        try:
            # Attempt retry with same parameters
            result = self._retry_function(context)
            self.recovery_stats['recovered_errors'] += 1
            return result
        except Exception as e:
            # Update context for next retry
            new_context = ErrorContext(
                error=e,
                severity=context.severity,
                timestamp=time.time(),
                function_name=context.function_name,
                args=context.args,
                kwargs=context.kwargs,
                retry_count=context.retry_count + 1
            )
            return self.handle_error(new_context)

    def _retry_function(self, context: ErrorContext) -> Any:
        """Retry the failed function."""
        # This is a placeholder - in real implementation,
        # we would need a way to re-execute the original function
        raise NotImplementedError("Function retry not implemented")

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        stats = self.recovery_stats.copy()
        if stats['total_errors'] > 0:
            stats['recovery_rate'] = stats['recovered_errors'] / stats['total_errors']
        else:
            stats['recovery_rate'] = 0.0
        return stats


class GracefulDegradation:
    """Graceful degradation implementation."""

    def __init__(self):
        self.fallback_functions: Dict[str, Callable] = {}
        self.degradation_stats = {
            'fallback_used': 0,
            'quality_reduction': 0.0
        }

    def register_fallback(self, function_name: str, fallback: Callable):
        """Register a fallback function."""
        self.fallback_functions[function_name] = fallback

    def can_recover(self, context: ErrorContext) -> bool:
        """Check if graceful degradation is possible."""
        return context.function_name in self.fallback_functions

    def handle_error(self, context: ErrorContext) -> Any:
        """Handle error with graceful degradation."""
        if not self.can_recover(context):
            raise context.error

        logging.warning(f"Using fallback for {context.function_name}")

        fallback = self.fallback_functions[context.function_name]
        try:
            # Use fallback function with reduced quality/functionality
            result = fallback(*context.args, **context.kwargs)
            self.degradation_stats['fallback_used'] += 1
            self.degradation_stats['quality_reduction'] += 0.1  # Assume 10% quality loss
            return result
        except Exception as e:
            logging.error(f"Fallback also failed for {context.function_name}: {e}")
            raise context.error

    def get_stats(self) -> Dict[str, Any]:
        """Get degradation statistics."""
        return self.degradation_stats.copy()


class UserConfigurableRecovery:
    """User-configurable recovery options."""

    def __init__(self):
        self.recovery_policies: Dict[str, RecoveryStrategy] = {}
        self.user_callbacks: Dict[str, Callable] = {}
        self.config_stats = {
            'policy_changes': 0,
            'user_interventions': 0
        }

    def set_recovery_policy(self, error_type: str, strategy: RecoveryStrategy):
        """Set recovery policy for specific error types."""
        self.recovery_policies[error_type] = strategy
        self.config_stats['policy_changes'] += 1

    def register_user_callback(self, error_type: str, callback: Callable):
        """Register user callback for error handling."""
        self.user_callbacks[error_type] = callback

    def can_recover(self, context: ErrorContext) -> bool:
        """Check if user-configured recovery is available."""
        error_type = type(context.error).__name__
        return error_type in self.recovery_policies

    def handle_error(self, context: ErrorContext) -> Any:
        """Handle error based on user configuration."""
        error_type = type(context.error).__name__

        if not self.can_recover(context):
            # No policy configured, raise the error
            raise context.error

        strategy = self.recovery_policies[error_type]

        if strategy == RecoveryStrategy.RETRY:
            return self._handle_retry(context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._handle_fallback(context)
        elif strategy == RecoveryStrategy.SKIP:
            return self._handle_skip(context)
        elif strategy == RecoveryStrategy.ABORT:
            return self._handle_abort(context)
        else:
            raise context.error

    def _handle_retry(self, context: ErrorContext) -> Any:
        """Handle retry strategy."""
        if context.retry_count >= 3:  # Max retries
            raise context.error

        # Use automatic recovery for retry
        auto_recovery = AutomaticErrorRecovery()
        return auto_recovery.handle_error(context)

    def _handle_fallback(self, context: ErrorContext) -> Any:
        """Handle fallback strategy."""
        # Invoke user callback if available
        error_type = type(context.error).__name__
        if error_type in self.user_callbacks:
            callback = self.user_callbacks[error_type]
            result = callback(context)
            self.config_stats['user_interventions'] += 1
            return result

        # Default fallback - return None or empty result
        logging.warning(f"Using default fallback for {context.function_name}")
        return None

    def _handle_skip(self, context: ErrorContext) -> Any:
        """Handle skip strategy."""
        logging.info(f"Skipping failed operation: {context.function_name}")
        return None

    def _handle_abort(self, context: ErrorContext) -> Any:
        """Handle abort strategy."""
        logging.critical(f"Aborting due to error in {context.function_name}")
        raise context.error

    def get_stats(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return self.config_stats.copy()


class ErrorRecoveryManager:
    """Manages different error recovery strategies."""

    def __init__(self):
        self.handlers = {
            'automatic': AutomaticErrorRecovery(),
            'graceful': GracefulDegradation(),
            'configurable': UserConfigurableRecovery()
        }
        self.default_handler = 'automatic'
        self.global_error_log: List[ErrorContext] = []

    def handle_error(self, error: Exception, function_name: str,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    *args, **kwargs) -> Any:
        """Handle an error using available strategies."""
        context = ErrorContext(
            error=error,
            severity=severity,
            timestamp=time.time(),
            function_name=function_name,
            args=args,
            kwargs=kwargs
        )

        self.global_error_log.append(context)

        # Try handlers in order of preference
        for handler_name in ['configurable', 'graceful', 'automatic']:
            handler = self.handlers[handler_name]
            if handler.can_recover(context):
                try:
                    return handler.handle_error(context)
                except Exception as recovery_error:
                    logging.error(f"Recovery failed with {handler_name}: {recovery_error}")
                    continue

        # All recovery attempts failed
        logging.critical(f"All recovery attempts failed for {function_name}")
        raise error

    def configure_handler(self, handler_type: str, **config):
        """Configure a specific error handler."""
        if handler_type not in self.handlers:
            raise ValueError(f"Unknown handler type: {handler_type}")

        handler = self.handlers[handler_type]

        if handler_type == 'configurable' and 'policies' in config:
            for error_type, strategy in config['policies'].items():
                handler.set_recovery_policy(error_type, strategy)

        if handler_type == 'graceful' and 'fallbacks' in config:
            for func_name, fallback in config['fallbacks'].items():
                handler.register_fallback(func_name, fallback)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        stats = {
            'total_errors': len(self.global_error_log),
            'error_by_severity': {},
            'error_by_type': {},
            'recovery_stats': {}
        }

        # Analyze error log
        for context in self.global_error_log:
            # By severity
            severity_name = context.severity.name
            stats['error_by_severity'][severity_name] = (
                stats['error_by_severity'].get(severity_name, 0) + 1
            )

            # By type
            error_type = type(context.error).__name__
            stats['error_by_type'][error_type] = (
                stats['error_by_type'].get(error_type, 0) + 1
            )

        # Get handler stats
        for name, handler in self.handlers.items():
            stats['recovery_stats'][name] = handler.get_stats()

        return stats


def error_recovery_decorator(recovery_manager: ErrorRecoveryManager,
                           severity: ErrorSeverity = ErrorSeverity.ERROR):
    """Decorator for automatic error recovery."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return recovery_manager.handle_error(
                    e, func.__name__, severity, *args, **kwargs
                )
        return wrapper
    return decorator


def async_error_recovery_decorator(recovery_manager: ErrorRecoveryManager,
                                 severity: ErrorSeverity = ErrorSeverity.ERROR):
    """Async decorator for automatic error recovery."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return recovery_manager.handle_error(
                    e, func.__name__, severity, *args, **kwargs
                )
        return wrapper
    return decorator
