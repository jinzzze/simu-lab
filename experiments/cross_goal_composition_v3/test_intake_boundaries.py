"""Protect raw-session path boundaries and prevent silent filename aliasing."""
from pathlib import Path
import pytest
from intake_recordings import inside_session


def test_parent_escape_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        inside_session(tmp_path/'X01','../X03/test.mp4')


def test_non_video_mapping_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        inside_session(tmp_path,'secrets.json')


def test_original_phone_filename_can_be_preserved(tmp_path):
    assert inside_session(tmp_path,'phone_original_001.MOV')==tmp_path.resolve()/'phone_original_001.MOV'
