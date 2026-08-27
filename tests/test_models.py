import numpy as np
import pandas as pd

from interaction_protocol.models import (
    COMPARISON_MODEL_FITTERS,
    MAIN_MODEL_NAME,
    MODEL_FITTERS,
    SENSITIVITY_MODEL_FITTERS,
    DebateExperiment,
    ModelMatrices,
    PaperModelConfig,
    fit_debate_random_intercept_model,
    fit_paper_model,
    model_coefficients,
    preprocess,
    resample_dilemmas,
)


def example_matrices():
    synchronous = pd.DataFrame(
        {
            "Agent_1_verdicts": [["NTA", "NTA"]],
            "Agent_2_verdicts": [["YTA", "NTA"]],
        }
    )
    round_robin = pd.DataFrame(
        {
            "Agent_1_verdicts": [["NTA"]],
            "Agent_2_verdicts": [["YTA"]],
        }
    )
    return preprocess(
        [DebateExperiment(synchronous, (0, 1))],
        [DebateExperiment(round_robin, (0, 1))],
    )


def test_preprocess_builds_shared_model_matrices():
    matrices = example_matrices()

    assert matrices.n_observations == 6
    assert matrices.n_models == 2
    assert matrices.n_dilemmas == 1
    assert matrices.n_verdicts == 5
    np.testing.assert_array_equal(matrices.y, [0, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(matrices.debate_idx, [0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(matrices.same_prev_mat[2], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(matrices.same_prev_mat[3], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(
        matrices.exposure_within_mat[-1], [1, 0, 0, 0, 0]
    )


def test_model_matrices_round_trip_through_legacy_dictionary():
    matrices = example_matrices()

    restored = ModelMatrices.from_legacy_dict(matrices.as_legacy_dict())

    for field in ModelMatrices.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(restored, field), getattr(matrices, field))


def test_paper_model_is_default_and_fits_common_matrices():
    matrices = example_matrices()
    config = PaperModelConfig(epochs=1, batch_size=6)

    fit = fit_paper_model(matrices, config, verbose=False)

    assert MAIN_MODEL_NAME == "paper"
    assert tuple(MODEL_FITTERS) == (MAIN_MODEL_NAME,)
    assert fit.name == MAIN_MODEL_NAME
    assert fit.n_observations == 6
    assert fit.epochs_fit == 1
    assert np.isfinite(fit.objective)


def test_dilemma_resampling_preserves_draw_multiplicity():
    matrices = example_matrices()

    resampled = resample_dilemmas(matrices, np.array([0, 0]))

    assert resampled.n_observations == matrices.n_observations * 2
    np.testing.assert_array_equal(
        resampled.y,
        np.repeat(matrices.y, 2),
    )


def test_every_comparison_model_supports_map_and_mle_fits():
    matrices = example_matrices()
    config = PaperModelConfig(epochs=2)

    assert tuple(COMPARISON_MODEL_FITTERS) == (
        "fixed_effects",
        "fixed_effects_inertia",
        "fixed_effects_inertia_global_conformity",
        "fixed_effects_inertia_previous_conformity",
        "fixed_effects_inertia_split_global_conformity",
        "paper",
        "paper_interactions",
    )
    for fitter in COMPARISON_MODEL_FITTERS.values():
        map_fit = fitter(matrices, config, penalized=True)
        mle_fit = fitter(matrices, config, penalized=False)

        assert map_fit.penalized is True
        assert mle_fit.penalized is False
        assert map_fit.n_parameters == mle_fit.n_parameters
        assert np.isfinite(map_fit.aic_at_fit)
        assert np.isfinite(mle_fit.aic_at_fit)


def test_random_intercept_is_map_only_and_reports_debate_effect_sd():
    matrices = example_matrices()
    config = PaperModelConfig(epochs=2)

    assert tuple(SENSITIVITY_MODEL_FITTERS) == ("paper_debate_random_intercept",)
    fit = fit_debate_random_intercept_model(matrices, config)
    coefficients = model_coefficients(fit)

    assert "debate_effect_sd" in set(coefficients.parameter)
    with np.testing.assert_raises_regex(ValueError, "MAP-only"):
        fit_debate_random_intercept_model(matrices, config, penalized=False)
