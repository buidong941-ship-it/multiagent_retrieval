"""Config package — expose Settings singleton."""

from config.settings import get_settings, settings

__all__ = ["settings", "get_settings"]
