"""Manufacturer naming policy shared with Mohamed's working Django service."""
from ..recognition.attributes.manufacturer.brands import MANUFACTURER_DISPLAY_NAMES

POLICY = 'recognized-name-v1'


def manufacturer_name(identifier):
    if identifier == 'unknown':
        return None
    # Recognition normalizes identifiers; its vocabulary is deliberately open.
    return MANUFACTURER_DISPLAY_NAMES.get(identifier, identifier)
