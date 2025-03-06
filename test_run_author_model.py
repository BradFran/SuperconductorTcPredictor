import numpy as np
import pytest
from run_author_model import run_model

def test_reproducibility():
    # Run the model twice with the same parameters.
    avg_rmse1, avg_r2_1 = run_model(n_runs=3, initial_seed=42)
    avg_rmse2, avg_r2_2 = run_model(n_runs=3, initial_seed=42)
    # They should be almost equal since the procedure is deterministic.
    np.testing.assert_allclose(avg_rmse1, avg_rmse2, rtol=1e-5)
    np.testing.assert_allclose(avg_r2_1, avg_r2_2, rtol=1e-5)

def test_return_types():
    avg_rmse, avg_r2 = run_model(n_runs=1, initial_seed=42)
    assert isinstance(avg_rmse, float)
    assert isinstance(avg_r2, float)

if __name__ == "__main__":
    pytest.main()
