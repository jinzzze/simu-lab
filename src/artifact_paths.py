"""Relocate historical Windows artifact paths without weakening content hash checks."""
from pathlib import PureWindowsPath

def historical_checkpoint_path_matches(state, expected_relative):
    """Compare identities inside the recorded project; callers must also verify SHA256.

    The frozen data_path identifies the old project root. It is never opened.
    Reproduction reads the expected checkpoint under the current project root.
    """
    try:
        data=PureWindowsPath(state['data_path'])
        checkpoint=PureWindowsPath(state['pretraining_path'])
        expected=PureWindowsPath(expected_relative)
        if not data.is_absolute() or not checkpoint.is_absolute() or expected.is_absolute():return False
        if data.parts[-3:-1]!=('data','processed'):return False
        if '..' in data.parts or '..' in checkpoint.parts or '..' in expected.parts:return False
        return checkpoint.relative_to(data.parents[2])==expected
    except (KeyError,ValueError,IndexError,TypeError):
        return False
