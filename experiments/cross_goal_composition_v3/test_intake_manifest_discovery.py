"""Keep cross-session duplicate checks working after derived review artifacts exist."""
import json
import pytest
from intake_recordings import prior_session_hashes


def save(root,name,value):
    folder=root/name
    folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps(value),encoding='utf-8')


def test_reviews_do_not_hide_prior_training_hashes(tmp_path):
    save(tmp_path,'raw_train',dict(status='inspected',session_id='X01',recordings=[{'sha256':'training_digest'}]))
    save(tmp_path,'visual_review',dict(status='frames_prepared',recordings=[{'raw_sha256':'training_digest'}]))
    save(tmp_path,'same_session',dict(status='inspected',session_id='X02',recordings=[{'sha256':'validation_digest'}]))
    hashes,sources=prior_session_hashes(tmp_path,'X02')
    assert hashes=={'training_digest'}
    assert list(sources)==['raw_train/manifest.json'] or list(sources)==['raw_train\\manifest.json']


def test_malformed_raw_intake_is_not_silently_ignored(tmp_path):
    save(tmp_path,'malformed',dict(status='inspected',recordings=[]))
    with pytest.raises(KeyError):
        prior_session_hashes(tmp_path,'X02')
