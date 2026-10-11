"""Real-data regression checks for split boundaries and profile provenance."""
import copy
import json
import unittest
import numpy as np
from common import ROOT,OLD,read,sha,check_sources
from prepare_study import transport_profile


class ProfileBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review=read(OLD/'data/processed/S01_trajectories_v1/S01_E01.reviewed.json')
        cls.cal=read(OLD/'data/processed/S01_calibration_v2/S01_E01.json')
        cls.rows=[json.loads(s) for s in (OLD/'data/processed/S01_trajectories_v1/S01_E01.frames.jsonl').read_text().splitlines()]

    def test_validation_rejected_at_review_and_frame_level(self):
        review=dict(self.review,split='val')
        with self.assertRaises(AssertionError):transport_profile(self.rows,review,self.cal,'P1')
        rows=copy.deepcopy(self.rows);rows[1]['split']='test'
        with self.assertRaises(AssertionError):transport_profile(rows,self.review,self.cal,'P1')

    def test_invalid_object_rows_never_become_constraints(self):
        result=transport_profile(self.rows,self.review,self.cal,'P1')
        self.assertTrue(all(self.rows[i]['object']['auxiliary_mask_candidate'] for i in result['valid_source_frame_indices']))
        rows=copy.deepcopy(self.rows)
        for row in rows[1:]:row['object']['auxiliary_mask_candidate']=False
        with self.assertRaises(AssertionError):transport_profile(rows,self.review,self.cal,'P1')

    def test_hand_predictions_do_not_change_progress(self):
        baseline=transport_profile(self.rows,self.review,self.cal,'P1')
        rows=copy.deepcopy(self.rows)
        for row in rows:row['hand']={'landmarks_normalized':[[[.99,.99,100]]*21]}
        changed=transport_profile(rows,self.review,self.cal,'P1')
        self.assertEqual(baseline['progress_knots'],changed['progress_knots'])

    def test_existing_s01_profile_recipe_preserved(self):
        # Bullet import occurs here only; this test process never imports Torch.
        from paired_sim import video_transport_profile
        prior=video_transport_profile(OLD,'S01_E01');current=transport_profile(self.rows,self.review,self.cal,'P1')
        np.testing.assert_array_equal(prior['progress_knots'],current['progress_knots'])
        self.assertEqual(prior['duration_s'],current['duration_s'])

    def test_frozen_sources_and_all_24_training_clips(self):
        bundle=read(ROOT/'data/processed/human_profiles_v1.json');check_sources(bundle['source_sha256'])
        self.assertEqual(len(bundle['profiles']),24)
        self.assertEqual({p['human_session'] for p in bundle['profiles']},{'S01','X01'})
        self.assertFalse(any('/X02' in p or '/X03' in p for p in bundle['source_sha256']))
        study=read(ROOT/'configs/study_v1.json')
        self.assertEqual(study['human_profiles_sha256'],sha(ROOT/'data/processed/human_profiles_v1.json'))
        self.assertTrue(all([p['start_id'],p['goal_id']] not in study['heldout_pairs'] for p in bundle['profiles']))
        self.assertTrue(set(c['scene_seed'] for c in study['train_cases']).isdisjoint(c['scene_seed'] for c in study['test_cases']))


if __name__=='__main__':unittest.main(verbosity=2)
