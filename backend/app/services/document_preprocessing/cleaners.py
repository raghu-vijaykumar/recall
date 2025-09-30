import re

from .config import PreprocessingConfig


class TextCleaner:
    """Handles basic text cleaning operations"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def clean_text(self, text: str) -> str:
        """Apply all configured text cleaning operations"""
        if not text:
            return ""

        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove email addresses
        if self.config.remove_emails:
            text = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text
            )

        # Remove HTML tags
        if self.config.remove_html:
            text = re.sub(r"<[^>]+>", "", text)

        # Normalize unicode
        if self.config.normalize_unicode:
            text = text.replace("\u00a0", " ")  # Non-breaking spaces
            text = re.sub(r"[\u2000-\u206f]", " ", text)  # Remove unicode separators

        # Convert to lowercase
        if self.config.lowercase:
            text = text.lower()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text
