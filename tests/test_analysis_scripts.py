import sys
from pathlib import Path

import numpy as np
import pandas as pd

from interaction_protocol.models import DebateExperiment, PaperModelConfig, preprocess

ANALYSIS_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(ANALYSIS_SCRIPTS))

from calibration_analysis import (  # noqa: E402
    calibration_bins,
    classwise_expected_calibration_error,
)
from heldout_accuracy import (  # noqa: E402
    fit_model_on_mask,
    make_folds,
    predict,
)


def validation_matrices():
    synchronous = pd.DataFrame(
        {
            "Agent_1_verdicts": [["NTA", "NTA"], ["YTA", "NTA"]],
            "Agent_2_verdicts": [["YTA", "NTA"], ["NTA", "YTA"]],
        }
    )
    return preprocess(
        [DebateExperiment(synchronous, (0, 1))],
        [],
    )


def test_heldout_dilemma_prediction_retains_unseen_fixed_effect_level():
    matrices = validation_matrices()
    train_mask = matrices.dilemma_idx == 0
    test_mask = ~train_mask

    model = fit_model_on_mask(
        matrices,
        train_mask,
        False,
        PaperModelConfig(epochs=1),
    )
    probabilities = predict(model, matrices, test_mask, "cpu", batch_size=8)

    assert model.phi_raw.shape[0] == matrices.n_dilemmas
    assert probabilities.shape == (int(test_mask.sum()), matrices.n_verdicts)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_debate_folds_do_not_split_debates_across_train_and_test():
    matrices = validation_matrices()

    for _, train_mask, test_mask in make_folds(matrices, 2, 0, "debate"):
        train_debates = set(matrices.debate_idx[train_mask])
        test_debates = set(matrices.debate_idx[test_mask])
        assert train_debates.isdisjoint(test_debates)


def test_classwise_calibration_bins_count_every_observation_per_class():
    y_true = np.array([0, 1])
    probabilities = np.array(
        [
            [0.6, 0.2, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.05, 0.05],
        ]
    )

    bins = calibration_bins(y_true, probabilities, n_bins=5)

    assert bins["n"].sum() == len(y_true) * probabilities.shape[1]
    assert np.isfinite(classwise_expected_calibration_error(bins))
