"""Protect substantive split and provenance boundaries, not document formatting."""
import pytest
import importlib.util
from common import verify_training_pairs,read,ROOT
spec=importlib.util.spec_from_file_location('composition_validator',ROOT/'validate_protocol.py')
validator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
validate=validator.validate


def test_heldout_pair_rejected_from_training():
    with pytest.raises(ValueError):
        verify_training_pairs([dict(split='train',start_id='P1',goal_id='G2')])


def test_validation_trajectory_rejected_from_training():
    with pytest.raises(ValueError):
        verify_training_pairs([dict(split='val',start_id='P1',goal_id='G1')])


def test_every_allowed_pair_accepted_and_all_four_holdouts_rejected():
    p=read(ROOT/'protocol.json')
    verify_training_pairs([dict(split='train',start_id=a,goal_id=b) for a,b in p['training_pairs']])
    for a,b in p['heldout_pairs']:
        with pytest.raises(ValueError):
            verify_training_pairs([dict(split='train',start_id=a,goal_id=b)])


def test_manifests_keep_sessions_and_sim_scene_splits_separate():
    assert validate()['status']=='passed'
