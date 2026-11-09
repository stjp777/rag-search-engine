"""Exception types raised by rag_search."""


class RagSearchError(Exception):
    """Base class for errors that should be reported to the user without a traceback."""


class DatasetError(RagSearchError):
    """The dataset file is missing or malformed."""


class IndexNotFoundError(RagSearchError):
    """A required cached index or embedding file does not exist."""


class LLMError(RagSearchError):
    """The language model could not be reached or returned unusable output."""
