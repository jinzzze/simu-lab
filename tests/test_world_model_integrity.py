import json
from pathlib import Path
import numpy as np
import pytest
from src.world_model.integrity import IntegrityError, require_hash, sha256, verify_receipt, validate_predictions
from src.world_model.numpy_model import MacroOutcomePredictor, features
ROOT=Path(__file__).resolve().parents[1]

def test_receipt_rejects_modified_missing_and_outside_files(tmp_path):
    f=tmp_path/'input';f.write_bytes(b'original');anchor=sha256(f)
    verify_receipt(tmp_path,{'input':anchor})
    f.write_bytes(b'changed')
    with pytest.raises(IntegrityError):require_hash(f,anchor)
    with pytest.raises(IntegrityError):verify_receipt(tmp_path,{'../escape':anchor})
    with pytest.raises(IntegrityError):verify_receipt(tmp_path,{})
    with pytest.raises(IntegrityError):require_hash(tmp_path/'missing',anchor)

@pytest.mark.parametrize('state,command',[(np.array([np.nan,0]),[0,0]),([0,0],[1]),([],[]),([1e300,0],[0,0])])
def test_invalid_external_state_rejected(state,command):
    with pytest.raises(ValueError):features(state,command)

def test_prediction_contract_rejects_nan_range_and_shape():
    a=np.zeros((3,2))
    for b in [np.full((3,2),np.nan),np.full((3,2),1.01),np.zeros((2,2))]:
        with pytest.raises(IntegrityError):validate_predictions(a,b,(3,2))

def test_model_schema_and_outputs(tmp_path):
    weights={'seeds':np.array([41]),'delta_scale_m':np.array(.15)}
    for layer,shape in [(0,(64,4)),(2,(64,64)),(4,(4,64))]:
        weights[f's41_w{layer}']=np.zeros(shape,dtype=np.float32)
        weights[f's41_b{layer}']=np.zeros(shape[0],dtype=np.float32)
    path=tmp_path/'model.npz';np.savez(path,**weights)
    pred=MacroOutcomePredictor(path)(np.array([.01,.02]),np.array([.01,.02]))
    np.testing.assert_allclose(pred['final_xy_m'],[.01,.02]);assert pred['success_probability']==.5
    weights['s41_w0'][0,0]=np.nan;np.savez(path,**weights)
    with pytest.raises(ValueError):MacroOutcomePredictor(path)
