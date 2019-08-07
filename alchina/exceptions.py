"""Custom exceptions."""


class InvalidInput(Exception):
    """Used when the user input is invalid."""

    pass


class NotFitted(Exception):
    """Used when the model must be fitted before using it."""

    pass


class NotBuilt(Exception):
    """Used when the object must be built before using it."""

    pass
