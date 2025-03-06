# test_run_author_model.py

import os
import sys
import numpy as np
import pytest
from run_author_model import run_model

def test_file_not_found(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    with pytest.raises(SystemExit) as e:
        run_model(n_runs=1, initial_seed=42, test_size=0.33)
    assert "Error" in str(e.value)

def test_printed_output(capsys):
    run_model(n_runs=1, initial_seed=42, test_size=0.33)
    captured = capsys.readouterr().out
    assert "Began" in captured
    assert "Finished" in captured

def test_reproducibility_and_return_types():
    # Run the model twice with a single run
    avg_rmse1, avg_r2_1 = run_model(n_runs=1, initial_seed=42, test_size=0.33)
    avg_rmse2, avg_r2_2 = run_model(n_runs=1, initial_seed=42, test_size=0.33)
    
    # Check reproducibility, the two runs should produce identical results
    np.testing.assert_allclose(avg_rmse1, avg_rmse2, rtol=1e-5)
    np.testing.assert_allclose(avg_r2_1, avg_r2_2, rtol=1e-5)
    
    # Check that the return values are floats and within expected range
    assert isinstance(avg_rmse1, float)
    assert isinstance(avg_r2_1, float)
    assert 0 < avg_rmse1 < 100  # adjust the upper bound as appropriate
    assert 0 <= avg_r2_1 <= 1

if __name__ == "__main__":
    pytest.main()
