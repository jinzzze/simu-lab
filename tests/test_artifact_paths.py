from src.artifact_paths import historical_checkpoint_path_matches

def test_relocation_preserves_identity_without_accessing_old_root():
    row={'data_path':r'Z:\old-project\data\processed\sim_adaptation_v1.npz','pretraining_path':r'Z:\old-project\artifacts\runs\imagenet_reference_initialization\R_seed7\checkpoint.pt'}
    expected='artifacts/runs/imagenet_reference_initialization/R_seed7/checkpoint.pt'
    assert historical_checkpoint_path_matches(row,expected)
    assert not historical_checkpoint_path_matches(row,expected.replace('seed7','seed17'))

def test_reject_cross_project_and_traversal():
    row={'data_path':r'Z:\old-project\data\processed\sim_adaptation_v1.npz','pretraining_path':r'Z:\other-project\artifacts\checkpoint.pt'}
    assert not historical_checkpoint_path_matches(row,'artifacts/checkpoint.pt')
    row['pretraining_path']=r'Z:\old-project\artifacts\..\artifacts\checkpoint.pt'
    assert not historical_checkpoint_path_matches(row,'artifacts/checkpoint.pt')
    assert not historical_checkpoint_path_matches({},'artifacts/checkpoint.pt')
