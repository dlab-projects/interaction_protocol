"""Preprocess debate histories and fit statistical models of verdict dynamics.

Every model in this module consumes the same :class:`ModelMatrices` object. The
preprocessing layer is deliberately separate from fitting so that model variants
can be compared on exactly the same observations and predictors.

The currently registered model is the multinomial model reported in the paper.
Additional model functions can be added to ``MODEL_FITTERS`` without changing
the data-loading script.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict, dataclass, replace
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


N_VERDICTS = 5
MODEL_INDICES = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2,
}
MODEL_NAMES = ("GPT-4.1", "Claude 3.7", "Gemini 2.0 Flash")
MAIN_MODEL_NAME = "paper"


@dataclass(frozen=True)
class DebateExperiment:
    """Pair a processed debate dataframe with its speaker/model ordering.

    ``model_indices`` maps ``Agent_1_verdicts``, ``Agent_2_verdicts``, and, when
    present, ``Agent_3_verdicts`` onto the stable indices in ``MODEL_INDICES``.
    The ordering is essential for round-robin data because it determines which
    agents were visible to later speakers within a round.
    """

    dataframe: pd.DataFrame
    model_indices: tuple[int, ...]


@dataclass(frozen=True)
class ModelMatrices:
    """Aligned NumPy arrays consumed by every debate-dynamics model.

    Each row represents one model verdict at one dilemma/round/speaker position.
    ``same_prev_mat`` is a one-hot indicator of that model's immediately prior
    verdict. ``exposure_prev_mat`` counts other agents' verdicts from completed
    rounds, while ``exposure_within_mat`` counts verdicts already expressed by
    earlier speakers in the current round. The three predictor matrices have one
    column per verdict class.

    ``dilemma_idx`` is shared across experiments because every experiment uses
    the same 1,000 dilemmas. ``debate_idx`` is unique to an experiment-dilemma
    pair and is therefore suitable for grouping observations from one debate.
    """

    y: np.ndarray
    model_idx: np.ndarray
    dilemma_idx: np.ndarray
    round_idx: np.ndarray
    speaker_pos: np.ndarray
    same_prev_mat: np.ndarray
    exposure_prev_mat: np.ndarray
    exposure_within_mat: np.ndarray
    is_sync: np.ndarray
    debate_idx: np.ndarray

    @property
    def n_observations(self) -> int:
        """Return the number of verdict observations (matrix rows)."""
        return len(self.y)

    @property
    def n_models(self) -> int:
        """Return the number of model identities represented in the matrices."""
        return int(self.model_idx.max()) + 1

    @property
    def n_dilemmas(self) -> int:
        """Return the number of shared dilemma fixed-effect levels."""
        return int(self.dilemma_idx.max()) + 1

    @property
    def n_verdicts(self) -> int:
        """Return the number of categorical verdict outcomes."""
        return self.same_prev_mat.shape[1]

    def as_legacy_dict(self) -> dict[str, np.ndarray]:
        """Return the key names expected by the original notebook utilities.

        New code should use the dataclass attributes. This adapter exists so the
        historical notebooks and review-response scripts remain runnable while
        their models are migrated into this module.
        """
        return {
            "y": self.y,
            "model_idx": self.model_idx,
            "dilemma_idx": self.dilemma_idx,
            "round_idx": self.round_idx,
            "speaker_pos": self.speaker_pos,
            "same_prev_mat": self.same_prev_mat,
            "E_prev_mat": self.exposure_prev_mat,
            "E_within_mat": self.exposure_within_mat,
            "is_sync": self.is_sync,
            "debate_idx": self.debate_idx,
        }

    @classmethod
    def from_legacy_dict(cls, data: dict[str, np.ndarray]) -> "ModelMatrices":
        """Construct matrices from the dictionary schema used by older scripts."""
        n_observations = len(data["y"])
        return cls(
            y=data["y"],
            model_idx=data["model_idx"],
            dilemma_idx=data["dilemma_idx"],
            round_idx=data["round_idx"],
            speaker_pos=data["speaker_pos"],
            same_prev_mat=data["same_prev_mat"],
            exposure_prev_mat=data["E_prev_mat"],
            exposure_within_mat=data["E_within_mat"],
            is_sync=data["is_sync"],
            debate_idx=data.get(
                "debate_idx", np.zeros(n_observations, dtype=np.int64)
            ),
        )


@dataclass(frozen=True)
class FitConfig:
    """Shared optimization settings for model-fitting functions.

    ``weight_decay`` is used by the historical AdamW fitting path. L-BFGS fits
    optimize the explicit MAP objective and therefore do not apply decoupled
    optimizer weight decay.
    """

    learning_rate: float = 2e-2
    epochs: int = 120
    batch_size: int = 4096
    seed: int = 0
    device: str = "cpu"
    tolerance: float = 1e-4
    early_stopping_patience: int = 10
    weight_decay: float = 1e-2


@dataclass(frozen=True)
class PaperModelConfig(FitConfig):
    """Optimization settings and Normal-prior scales for the paper model.

    The sigma fields are standard deviations of zero-centered Normal
    priors. ``theta`` and ``phi`` are model-by-verdict and dilemma-by-verdict
    fixed effects; ``alpha`` is inertia; the gamma terms are previous-round and
    within-round conformity.
    """

    sigma_theta: float = 1.0
    sigma_phi: float = 1.0
    sigma_alpha: float = 0.5
    sigma_gamma_prev: float = 0.5
    sigma_gamma_within: float = 0.5
    sigma_interaction: float = 0.5
    sigma_debate: float = 1.0


@dataclass
class ModelFit:
    """A fitted torch module plus optimizer-independent fit metadata."""

    name: str
    model: nn.Module
    config: FitConfig
    n_observations: int
    n_parameters: int
    epochs_fit: int
    negative_log_likelihood: float
    objective: float
    penalized: bool = True

    @property
    def log_likelihood(self) -> float:
        """Return the unpenalized summed log-likelihood at the fitted coefficients."""
        return -self.negative_log_likelihood * self.n_observations

    @property
    def aic_at_fit(self) -> float:
        """Evaluate ``2k - 2 log L`` at this fit's coefficients.

        This is standard AIC only when ``penalized`` is false and the fit is the
        unpenalized MLE. For MAP fits it is reported as ``aic_at_map``.
        """
        return 2 * self.n_parameters - 2 * self.log_likelihood

    def metadata(self) -> dict[str, object]:
        """Flatten fit metrics and configuration into a serializable mapping."""
        return {
            "model": self.name,
            "n": self.n_observations,
            "k": self.n_parameters,
            "epochs_fit": self.epochs_fit,
            "negative_log_likelihood": self.negative_log_likelihood,
            "log_likelihood": self.log_likelihood,
            "objective": self.objective,
            "penalized": self.penalized,
            "aic_at_fit": self.aic_at_fit,
            **asdict(self.config),
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Coefficient draws from independent dilemma-cluster bootstrap fits.

    Every array has shape ``(n_bootstrap, n_models)``. Rows are ordered by seed;
    columns follow ``MODEL_NAMES``.
    """

    alpha: np.ndarray
    gamma_prev: np.ndarray
    gamma_within: np.ndarray

    def summary(self) -> pd.DataFrame:
        """Return medians and 95% percentile intervals on log-odds and OR scales.

        Exponentiation is monotonic, so exponentiating the coefficient interval
        endpoints gives the corresponding percentile interval for odds ratios.
        """
        rows = []
        for parameter, draws in (
            ("alpha", self.alpha),
            ("gamma_prev", self.gamma_prev),
            ("gamma_within", self.gamma_within),
        ):
            intervals = np.percentile(draws, [2.5, 50, 97.5], axis=0)
            for model_name, lower, median, upper in zip(MODEL_NAMES, *intervals):
                rows.append(
                    {
                        "model": model_name,
                        "parameter": parameter,
                        "estimate": float(median),
                        "ci_low": float(lower),
                        "ci_high": float(upper),
                        "odds_ratio": float(np.exp(median)),
                        "odds_ratio_ci_low": float(np.exp(lower)),
                        "odds_ratio_ci_high": float(np.exp(upper)),
                    }
                )
        return pd.DataFrame(rows)


def verdict2num(verdict: str) -> int:
    """Map an AITA verdict label to its stable multinomial outcome index.

    ``YWBTA`` is treated as the prospective form of ``YTA`` and mapped to the
    same class. Unknown labels fail loudly so malformed verdict histories do not
    silently enter the model.
    """
    if verdict == "NTA":
        return 0
    if verdict in {"YTA", "YWBTA"}:
        return 1
    if verdict == "NAH":
        return 2
    if verdict == "ESH":
        return 3
    if verdict == "INFO":
        return 4
    raise ValueError(f"Unknown verdict: {verdict}")


def make_verdict2num_mapper(mapper):
    """Normalize a legacy verdict mapper supplied as a callable or dictionary."""
    if callable(mapper):
        return mapper
    if isinstance(mapper, dict):
        return lambda verdict: mapper[verdict]
    raise TypeError("verdict2num must be dict or callable")


def df_to_verdict_lists(dataframe: pd.DataFrame, n_agents: int) -> list[list]:
    """Extract one dilemma-by-round verdict list for each agent column.

    The returned outer list is ordered by speaking position. Each element is a
    list over dilemmas, whose items are the verdict sequence for that dilemma.
    """
    if n_agents not in {2, 3}:
        raise ValueError("n_agents must be 2 or 3")
    return [
        dataframe[f"Agent_{agent_index}_verdicts"].tolist()
        for agent_index in range(1, n_agents + 1)
    ]


def _one_hot(index: int) -> np.ndarray:
    """Construct a float32 one-hot row for one verdict class."""
    row = np.zeros(N_VERDICTS, dtype=np.float32)
    row[index] = 1.0
    return row


def _is_missing(value, missing_tokens: tuple[object, ...]) -> bool:
    """Recognize absent verdicts without treating valid strings as missing."""
    return (
        value is None
        or (isinstance(value, float) and np.isnan(value))
        or value in missing_tokens
    )


def convert_synchronous(
    verdict_lists,
    model_indices,
    verdict_mapper=verdict2num,
    missing_tokens=("", None),
) -> dict[str, np.ndarray]:
    """Convert synchronous verdict histories into observation-level matrices.

    Synchronous agents cannot observe one another inside the current round, so
    ``E_within_mat`` is always zero. At the end of each round, all present
    verdicts are added simultaneously to the history used by the next round.

    ``E_prev_mat`` excludes the focal model's own past verdicts. This prevents
    the conformity predictor from duplicating ``same_prev_mat``, which captures
    the focal model's own immediately preceding verdict.
    """
    n_dilemmas = len(verdict_lists[0])
    y, model_idx, dilemma_idx, round_idx, speaker_pos = [], [], [], [], []
    same_prev_rows, exposure_prev_rows, exposure_within_rows = [], [], []

    # These state tables are maintained separately for every model-dilemma pair.
    # ``total_previous`` counts all verdicts in completed rounds, while
    # ``self_cumulative`` lets us subtract the focal model's own contributions.
    last_label = {
        (model, dilemma): None
        for dilemma in range(n_dilemmas)
        for model in model_indices
    }
    self_cumulative = {
        (model, dilemma): np.zeros(N_VERDICTS, dtype=np.float32)
        for dilemma in range(n_dilemmas)
        for model in model_indices
    }
    total_previous = {
        dilemma: np.zeros(N_VERDICTS, dtype=np.float32)
        for dilemma in range(n_dilemmas)
    }

    for dilemma in range(n_dilemmas):
        for round_number in range(len(verdict_lists[0][dilemma])):
            # Accumulate the round first, then publish all verdicts together.
            # Updating history inside this loop would incorrectly make the
            # synchronous protocol behave like round-robin.
            round_sum = np.zeros(N_VERDICTS, dtype=np.float32)
            present_agents = []
            for agent, model in enumerate(model_indices):
                verdict_string = verdict_lists[agent][dilemma][round_number]
                if _is_missing(verdict_string, missing_tokens):
                    continue
                verdict = verdict_mapper(verdict_string)

                y.append(verdict)
                model_idx.append(model)
                dilemma_idx.append(dilemma)
                round_idx.append(round_number + 1)
                speaker_pos.append(0)
                # Inertia is represented as a one-hot vector so the coefficient
                # is added only to the logit of the model's prior verdict.
                previous = last_label[(model, dilemma)]
                same_prev_rows.append(
                    np.zeros(N_VERDICTS, dtype=np.float32)
                    if previous is None
                    else _one_hot(previous)
                )
                # Prior-round conformity counts other agents' verdicts by class
                # across every completed round of this dilemma.
                exposure_previous = (
                    total_previous[dilemma] - self_cumulative[(model, dilemma)]
                )
                exposure_previous[exposure_previous < 0] = 0.0
                exposure_prev_rows.append(exposure_previous)
                exposure_within_rows.append(np.zeros(N_VERDICTS, dtype=np.float32))
                round_sum += _one_hot(verdict)
                present_agents.append((model, verdict))

            total_previous[dilemma] += round_sum
            for model, verdict in present_agents:
                self_cumulative[(model, dilemma)] += _one_hot(verdict)
                last_label[(model, dilemma)] = verdict

    return _matrix_part(
        y,
        model_idx,
        dilemma_idx,
        round_idx,
        speaker_pos,
        same_prev_rows,
        exposure_prev_rows,
        exposure_within_rows,
        is_sync=True,
    )


def convert_round_robin(
    verdict_lists,
    model_indices,
    verdict_mapper=verdict2num,
    missing_tokens=("", None),
) -> dict[str, np.ndarray]:
    """Convert sequential round-robin verdict histories into model matrices.

    Unlike synchronous debate, later speakers can observe verdicts already
    expressed in the current round. ``within_counts`` is therefore copied into
    each observation before adding that speaker's verdict. Completed-round
    exposure and own-prior-verdict inertia use the same definitions as the
    synchronous converter.
    """
    n_agents = len(verdict_lists)
    n_dilemmas = len(verdict_lists[0])
    y, model_idx, dilemma_idx, round_idx, speaker_pos = [], [], [], [], []
    same_prev_rows, exposure_prev_rows, exposure_within_rows = [], [], []

    last_label = {
        (model, dilemma): None
        for dilemma in range(n_dilemmas)
        for model in model_indices
    }
    self_cumulative = {
        (model, dilemma): np.zeros(N_VERDICTS, dtype=np.float32)
        for dilemma in range(n_dilemmas)
        for model in model_indices
    }
    total_previous = {
        dilemma: np.zeros(N_VERDICTS, dtype=np.float32)
        for dilemma in range(n_dilemmas)
    }

    for dilemma in range(n_dilemmas):
        n_rounds = max(len(verdict_lists[agent][dilemma]) for agent in range(n_agents))
        for round_number in range(n_rounds):
            # This counter is reset at the beginning of every round. It grows in
            # speaking order and is observed before the current verdict is added.
            within_counts = np.zeros(N_VERDICTS, dtype=np.float32)
            round_sum = np.zeros(N_VERDICTS, dtype=np.float32)
            present_agents = []
            for position, (agent, model) in enumerate(
                zip(range(n_agents), model_indices), start=1
            ):
                if round_number >= len(verdict_lists[agent][dilemma]):
                    continue
                verdict_string = verdict_lists[agent][dilemma][round_number]
                if _is_missing(verdict_string, missing_tokens):
                    continue
                verdict = verdict_mapper(verdict_string)

                y.append(verdict)
                model_idx.append(model)
                dilemma_idx.append(dilemma)
                round_idx.append(round_number + 1)
                speaker_pos.append(position)
                previous = last_label[(model, dilemma)]
                same_prev_rows.append(
                    np.zeros(N_VERDICTS, dtype=np.float32)
                    if previous is None
                    else _one_hot(previous)
                )
                exposure_previous = (
                    total_previous[dilemma] - self_cumulative[(model, dilemma)]
                )
                exposure_previous[exposure_previous < 0] = 0.0
                exposure_prev_rows.append(exposure_previous)
                # Copying is required because ``within_counts`` is mutated for
                # later speakers after this observation has been recorded.
                exposure_within_rows.append(within_counts.copy())
                within_counts += _one_hot(verdict)
                round_sum += _one_hot(verdict)
                present_agents.append((model, verdict))

            total_previous[dilemma] += round_sum
            for model, verdict in present_agents:
                self_cumulative[(model, dilemma)] += _one_hot(verdict)
                last_label[(model, dilemma)] = verdict

    return _matrix_part(
        y,
        model_idx,
        dilemma_idx,
        round_idx,
        speaker_pos,
        same_prev_rows,
        exposure_prev_rows,
        exposure_within_rows,
        is_sync=False,
    )


def _matrix_part(
    y,
    model_idx,
    dilemma_idx,
    round_idx,
    speaker_pos,
    same_prev_rows,
    exposure_prev_rows,
    exposure_within_rows,
    *,
    is_sync: bool,
) -> dict[str, np.ndarray]:
    """Materialize one protocol/experiment's accumulated Python rows as arrays."""
    n_rows = len(y)
    return {
        "y": np.asarray(y, dtype=np.int64),
        "model_idx": np.asarray(model_idx, dtype=np.int64),
        "dilemma_idx": np.asarray(dilemma_idx, dtype=np.int64),
        "round_idx": np.asarray(round_idx, dtype=np.int32),
        "speaker_pos": np.asarray(speaker_pos, dtype=np.int32),
        "same_prev_mat": _stack_rows(same_prev_rows),
        "E_prev_mat": _stack_rows(exposure_prev_rows),
        "E_within_mat": _stack_rows(exposure_within_rows),
        "is_sync": np.full(n_rows, is_sync, dtype=bool),
    }


def _stack_rows(rows) -> np.ndarray:
    """Stack verdict-vector rows while preserving a valid empty matrix shape."""
    if rows:
        return np.vstack(rows).astype(np.float32)
    return np.zeros((0, N_VERDICTS), dtype=np.float32)


def _concatenate_parts(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate aligned experiment matrices along their observation axis."""
    if not parts:
        raise ValueError("At least one debate experiment is required")
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    }


def preprocess(
    synchronous: Sequence[DebateExperiment],
    round_robin: Sequence[DebateExperiment],
    *,
    missing_tokens=("", None),
) -> ModelMatrices:
    """Preprocess all debate experiments into one shared matrix bundle.

    Parameters
    ----------
    synchronous
        Experiments in which all agents answer in parallel within a round.
    round_robin
        Experiments in which agents answer sequentially within a round.
    missing_tokens
        Additional scalar values that indicate an absent verdict.

    Returns
    -------
    ModelMatrices
        Vertically concatenated observations with a shared dilemma index and a
        globally unique debate index. The shared dilemma index lets ``phi``
        represent the same underlying AITA scenario across experiments. The
        debate index distinguishes each experiment-specific run of that dilemma.
    """
    parts = []
    debate_offset = 0

    for experiment, synchronous_protocol in [
        *((experiment, True) for experiment in synchronous),
        *((experiment, False) for experiment in round_robin),
    ]:
        # The protocol-specific converters share output keys, allowing every
        # downstream model to consume one stable matrix schema.
        verdict_lists = df_to_verdict_lists(
            experiment.dataframe, len(experiment.model_indices)
        )
        converter = convert_synchronous if synchronous_protocol else convert_round_robin
        part = converter(
            verdict_lists,
            experiment.model_indices,
            verdict2num,
            missing_tokens=missing_tokens,
        )
        # Dilemma IDs intentionally remain shared; debate IDs must not collide
        # across experiments because each file is a separate debate run.
        part["debate_idx"] = part["dilemma_idx"].astype(np.int64) + debate_offset
        if len(part["dilemma_idx"]):
            debate_offset += int(part["dilemma_idx"].max()) + 1
        parts.append(part)

    data = _concatenate_parts(parts)
    return ModelMatrices(
        y=data["y"],
        model_idx=data["model_idx"],
        dilemma_idx=data["dilemma_idx"],
        round_idx=data["round_idx"],
        speaker_pos=data["speaker_pos"],
        same_prev_mat=data["same_prev_mat"],
        exposure_prev_mat=data["E_prev_mat"],
        exposure_within_mat=data["E_within_mat"],
        is_sync=data["is_sync"],
        debate_idx=data["debate_idx"],
    )


def build_all_data(
    sync_exps,
    sync_model_ids,
    rr_exps,
    rr_model_ids,
    missing_tokens=("", None),
) -> dict[str, np.ndarray]:
    """Adapt the historical four-list interface to :func:`preprocess`.

    This wrapper returns a dictionary only for compatibility with notebooks and
    scripts written before ``ModelMatrices`` was introduced.
    """
    matrices = preprocess(
        [
            DebateExperiment(dataframe, tuple(model_indices))
            for dataframe, model_indices in zip(sync_exps, sync_model_ids)
        ],
        [
            DebateExperiment(dataframe, tuple(model_indices))
            for dataframe, model_indices in zip(rr_exps, rr_model_ids)
        ],
        missing_tokens=missing_tokens,
    )
    return matrices.as_legacy_dict()


class PaperModel(nn.Module):
    """Published multinomial model of model-specific inertia and conformity.

    For observation ``i`` and candidate verdict ``v``, the unnormalized logit is
    the sum of model-by-verdict and dilemma-by-verdict fixed effects, an inertia
    term for repeating the focal model's previous verdict, a previous-round
    conformity term, and a within-round conformity term. The latter three
    coefficients are model-specific.

    ``theta_raw`` and ``phi_raw`` are centered across verdicts in ``forward``.
    Softmax probabilities are invariant to a common additive shift, so centering
    establishes an identifiable location for these fixed effects without using
    an omitted reference verdict.
    """

    def __init__(self, n_models: int, n_dilemmas: int, n_verdicts: int = N_VERDICTS):
        """Initialize all fixed effects and behavioral coefficients at zero."""
        super().__init__()
        self.theta_raw = nn.Parameter(torch.zeros(n_models, n_verdicts))
        self.phi_raw = nn.Parameter(torch.zeros(n_dilemmas, n_verdicts))
        self.alpha = nn.Parameter(torch.zeros(n_models))
        self.gamma_prev_m = nn.Parameter(torch.zeros(n_models))
        self.gamma_within_m = nn.Parameter(torch.zeros(n_models))

    def forward(
        self,
        model_idx,
        dilemma_idx,
        same_prev,
        exposure_prev,
        exposure_within,
    ):
        """Return log-probabilities for every verdict class in each observation."""
        # Center only the verdict-specific fixed effects. Alpha and gamma already
        # operate on class-specific predictor vectors and need no location
        # constraint.
        theta = self.theta_raw - self.theta_raw.mean(dim=1, keepdim=True)
        phi = self.phi_raw - self.phi_raw.mean(dim=1, keepdim=True)
        logits = (
            theta[model_idx]
            + phi[dilemma_idx]
            + self.alpha[model_idx].unsqueeze(1) * same_prev
            + self.gamma_prev_m[model_idx].unsqueeze(1) * exposure_prev
            + self.gamma_within_m[model_idx].unsqueeze(1) * exposure_within
        )
        return torch.log_softmax(logits, dim=1)


@dataclass(frozen=True)
class ModelSpecification:
    """Declare which behavioral terms appear in a comparison model.

    ``previous`` and ``within`` may be ``None``, ``"global"``, or
    ``"per_model"``. A blended conformity term uses one coefficient for the
    sum of previous-round and within-round verdict counts. Interactions multiply
    the inertia indicator by the corresponding conformity exposure.
    """

    name: str
    inertia: bool = False
    blended_conformity: bool = False
    previous: str | None = None
    within: str | None = None
    interaction: str | None = None
    debate_random_intercept: bool = False

    def __post_init__(self) -> None:
        """Reject internally inconsistent specifications at construction time."""
        valid_scopes = {None, "global", "per_model"}
        if self.previous not in valid_scopes or self.within not in valid_scopes:
            raise ValueError("Conformity scope must be global, per_model, or None")
        if self.interaction not in valid_scopes:
            raise ValueError("Interaction scope must be global, per_model, or None")
        if self.interaction and not self.inertia:
            raise ValueError("Inertia-conformity interactions require inertia")
        if self.interaction and (self.previous is None or self.within is None):
            raise ValueError("Interactions require both conformity terms")


class ComparisonModel(nn.Module):
    """Multinomial verdict model assembled from a :class:`ModelSpecification`.

    The fixed effects and data inputs are common to all variants. This class is
    intentionally internal to the fitting API: callers use the named public fit
    functions below, which make the scientific meaning of each model explicit.
    """

    def __init__(
        self,
        n_models: int,
        n_dilemmas: int,
        n_verdicts: int,
        specification: ModelSpecification,
        *,
        n_debates: int | None = None,
    ) -> None:
        """Initialize the selected fixed, behavioral, and debate-level effects."""
        super().__init__()
        self.specification = specification
        self.theta_raw = nn.Parameter(torch.zeros(n_models, n_verdicts))
        self.phi_raw = nn.Parameter(torch.zeros(n_dilemmas, n_verdicts))

        if specification.inertia:
            self.alpha = nn.Parameter(torch.zeros(n_models))
        if specification.blended_conformity:
            self.gamma = nn.Parameter(torch.tensor(0.0))
        if specification.previous == "global":
            self.gamma_prev = nn.Parameter(torch.tensor(0.0))
        elif specification.previous == "per_model":
            self.gamma_prev_m = nn.Parameter(torch.zeros(n_models))
        if specification.within == "global":
            self.gamma_within = nn.Parameter(torch.tensor(0.0))
        elif specification.within == "per_model":
            self.gamma_within_m = nn.Parameter(torch.zeros(n_models))
        if specification.interaction == "global":
            self.delta_prev = nn.Parameter(torch.tensor(0.0))
            self.delta_within = nn.Parameter(torch.tensor(0.0))
        elif specification.interaction == "per_model":
            self.delta_prev_m = nn.Parameter(torch.zeros(n_models))
            self.delta_within_m = nn.Parameter(torch.zeros(n_models))

        if specification.debate_random_intercept:
            if n_debates is None:
                raise ValueError("n_debates is required for a debate random effect")
            # A separate verdict contrast is needed: one scalar added to every
            # logit would cancel under softmax and have no statistical effect.
            self.debate_raw = nn.Parameter(torch.zeros(n_debates, n_verdicts))

    def forward(
        self,
        model_idx,
        dilemma_idx,
        same_prev,
        exposure_prev,
        exposure_within,
        debate_idx=None,
    ):
        """Return class log-probabilities under the selected model terms."""
        theta = self.theta_raw - self.theta_raw.mean(dim=1, keepdim=True)
        phi = self.phi_raw - self.phi_raw.mean(dim=1, keepdim=True)
        logits = theta[model_idx] + phi[dilemma_idx]

        if self.specification.debate_random_intercept:
            if debate_idx is None:
                raise ValueError("debate_idx is required for a debate random effect")
            debate = self.debate_raw - self.debate_raw.mean(dim=1, keepdim=True)
            logits = logits + debate[debate_idx]
        if self.specification.inertia:
            logits = logits + self.alpha[model_idx].unsqueeze(1) * same_prev
        if self.specification.blended_conformity:
            logits = logits + self.gamma * (exposure_prev + exposure_within)
        if self.specification.previous == "global":
            logits = logits + self.gamma_prev * exposure_prev
        elif self.specification.previous == "per_model":
            logits = (
                logits
                + self.gamma_prev_m[model_idx].unsqueeze(1) * exposure_prev
            )
        if self.specification.within == "global":
            logits = logits + self.gamma_within * exposure_within
        elif self.specification.within == "per_model":
            logits = (
                logits
                + self.gamma_within_m[model_idx].unsqueeze(1) * exposure_within
            )
        if self.specification.interaction == "global":
            logits = (
                logits
                + self.delta_prev * (same_prev * exposure_prev)
                + self.delta_within * (same_prev * exposure_within)
            )
        elif self.specification.interaction == "per_model":
            logits = (
                logits
                + self.delta_prev_m[model_idx].unsqueeze(1)
                * (same_prev * exposure_prev)
                + self.delta_within_m[model_idx].unsqueeze(1)
                * (same_prev * exposure_within)
            )
        return torch.log_softmax(logits, dim=1)


FIXED_EFFECTS_SPEC = ModelSpecification("fixed_effects")
INERTIA_SPEC = ModelSpecification("fixed_effects_inertia", inertia=True)
GLOBAL_CONFORMITY_SPEC = ModelSpecification(
    "fixed_effects_inertia_global_conformity",
    inertia=True,
    blended_conformity=True,
)
PREVIOUS_CONFORMITY_SPEC = ModelSpecification(
    "fixed_effects_inertia_previous_conformity",
    inertia=True,
    previous="global",
)
SPLIT_GLOBAL_CONFORMITY_SPEC = ModelSpecification(
    "fixed_effects_inertia_split_global_conformity",
    inertia=True,
    previous="global",
    within="global",
)
PAPER_COMPARISON_SPEC = ModelSpecification(
    MAIN_MODEL_NAME,
    inertia=True,
    previous="per_model",
    within="per_model",
)
INTERACTION_SPEC = ModelSpecification(
    "paper_interactions",
    inertia=True,
    previous="per_model",
    within="per_model",
    interaction="per_model",
)
DEBATE_RANDOM_INTERCEPT_SPEC = ModelSpecification(
    "paper_debate_random_intercept",
    inertia=True,
    previous="per_model",
    within="per_model",
    debate_random_intercept=True,
)


def _identifiable_parameter_count(model: ComparisonModel) -> int:
    """Count estimable verdict contrasts and behavioral coefficients.

    Each centered verdict-vector effect contributes ``n_verdicts - 1`` degrees
    of freedom. This matches the convention used for the review-response AIC
    table. The debate effects use the same centered-contrast accounting, though
    that sensitivity model is deliberately excluded from standard AIC ranking.
    """
    n_models, n_verdicts = model.theta_raw.shape
    n_dilemmas = model.phi_raw.shape[0]
    count = (n_models + n_dilemmas) * (n_verdicts - 1)
    for name, parameter in model.named_parameters():
        if name in {"theta_raw", "phi_raw"}:
            continue
        if name == "debate_raw":
            count += parameter.shape[0] * (parameter.shape[1] - 1)
        else:
            count += parameter.numel()
    return int(count)


def _fit_comparison_model(
    matrices: ModelMatrices,
    specification: ModelSpecification,
    config: PaperModelConfig | None = None,
    *,
    penalized: bool = True,
    verbose: bool = False,
) -> ModelFit:
    """Fit one declared model by full-batch L-BFGS.

    With ``penalized=True``, the objective is mean negative log-likelihood plus
    explicit zero-centered Gaussian penalties for every fitted coefficient.
    With ``penalized=False``, the optimizer targets likelihood alone and the
    resulting fit supports standard AIC. No optimizer-level weight decay is
    used in either case, so the penalty is completely explicit and auditable.
    """
    config = config or PaperModelConfig()
    if specification.debate_random_intercept and not penalized:
        raise ValueError(
            "The debate random-intercept sensitivity model is MAP-only; "
            "it is excluded from the standard AIC comparison."
        )
    torch.manual_seed(config.seed)
    device = config.device
    y = torch.as_tensor(matrices.y, dtype=torch.long, device=device)
    model_idx = torch.as_tensor(matrices.model_idx, dtype=torch.long, device=device)
    dilemma_idx = torch.as_tensor(matrices.dilemma_idx, dtype=torch.long, device=device)
    same_prev = torch.as_tensor(matrices.same_prev_mat, dtype=torch.float32, device=device)
    exposure_prev = torch.as_tensor(
        matrices.exposure_prev_mat, dtype=torch.float32, device=device
    )
    exposure_within = torch.as_tensor(
        matrices.exposure_within_mat, dtype=torch.float32, device=device
    )
    debate_idx = torch.as_tensor(matrices.debate_idx, dtype=torch.long, device=device)
    n_debates = int(matrices.debate_idx.max()) + 1
    model = ComparisonModel(
        matrices.n_models,
        matrices.n_dilemmas,
        matrices.n_verdicts,
        specification,
        n_debates=n_debates if specification.debate_random_intercept else None,
    ).to(device)
    criterion = nn.NLLLoss(reduction="mean")

    def prior_term():
        """Return explicit L2 penalties on the mean-loss scale."""
        if not penalized:
            return torch.zeros((), dtype=torch.float32, device=device)
        penalty = (
            model.theta_raw.pow(2).sum() / (2 * config.sigma_theta**2)
            + model.phi_raw.pow(2).sum() / (2 * config.sigma_phi**2)
        )
        if hasattr(model, "alpha"):
            penalty = penalty + model.alpha.pow(2).sum() / (2 * config.sigma_alpha**2)
        for name in ("gamma", "gamma_prev", "gamma_prev_m"):
            if hasattr(model, name):
                parameter = getattr(model, name)
                penalty = penalty + parameter.pow(2).sum() / (
                    2 * config.sigma_gamma_prev**2
                )
        for name in ("gamma_within", "gamma_within_m"):
            if hasattr(model, name):
                parameter = getattr(model, name)
                penalty = penalty + parameter.pow(2).sum() / (
                    2 * config.sigma_gamma_within**2
                )
        for name in ("delta_prev", "delta_within", "delta_prev_m", "delta_within_m"):
            if hasattr(model, name):
                parameter = getattr(model, name)
                penalty = penalty + parameter.pow(2).sum() / (
                    2 * config.sigma_interaction**2
                )
        if hasattr(model, "debate_raw"):
            penalty = penalty + model.debate_raw.pow(2).sum() / (
                2 * config.sigma_debate**2
            )
        return penalty / matrices.n_observations

    def loss_value():
        """Evaluate the differentiable full-data fitting objective."""
        log_probabilities = model(
            model_idx,
            dilemma_idx,
            same_prev,
            exposure_prev,
            exposure_within,
            debate_idx if specification.debate_random_intercept else None,
        )
        return criterion(log_probabilities, y) + prior_term()

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=config.epochs,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        """Supply L-BFGS with a fresh objective and gradient."""
        optimizer.zero_grad()
        loss = loss_value()
        loss.backward()
        return loss

    optimizer.step(closure)
    model.eval()
    with torch.no_grad():
        log_probabilities = model(
            model_idx,
            dilemma_idx,
            same_prev,
            exposure_prev,
            exposure_within,
            debate_idx if specification.debate_random_intercept else None,
        )
        negative_log_likelihood = criterion(log_probabilities, y).item()
        objective = negative_log_likelihood + float(prior_term())
    state = optimizer.state.get(model.theta_raw, {})
    iterations = int(state.get("n_iter", config.epochs))
    if verbose:
        fit_kind = "MAP" if penalized else "MLE"
        print(
            f"{specification.name} {fit_kind}: objective={objective:.6f} "
            f"nll={negative_log_likelihood:.6f} iterations={iterations}"
        )
    return ModelFit(
        name=specification.name,
        model=model,
        config=config,
        n_observations=matrices.n_observations,
        n_parameters=_identifiable_parameter_count(model),
        epochs_fit=iterations,
        negative_log_likelihood=negative_log_likelihood,
        objective=objective,
        penalized=penalized,
    )


def fit_fixed_effects_model(matrices, config=None, *, penalized=True, verbose=False):
    """Fit model- and dilemma-specific verdict fixed effects only."""
    return _fit_comparison_model(
        matrices, FIXED_EFFECTS_SPEC, config, penalized=penalized, verbose=verbose
    )


def fit_inertia_model(matrices, config=None, *, penalized=True, verbose=False):
    """Fit fixed effects plus model-specific persistence of the prior verdict."""
    return _fit_comparison_model(
        matrices, INERTIA_SPEC, config, penalized=penalized, verbose=verbose
    )


def fit_global_conformity_model(
    matrices, config=None, *, penalized=True, verbose=False
):
    """Fit inertia plus one conformity coefficient for all exposure counts."""
    return _fit_comparison_model(
        matrices,
        GLOBAL_CONFORMITY_SPEC,
        config,
        penalized=penalized,
        verbose=verbose,
    )


def fit_previous_round_conformity_model(
    matrices, config=None, *, penalized=True, verbose=False
):
    """Fit inertia plus a global previous-round conformity coefficient."""
    return _fit_comparison_model(
        matrices,
        PREVIOUS_CONFORMITY_SPEC,
        config,
        penalized=penalized,
        verbose=verbose,
    )


def fit_split_global_conformity_model(
    matrices, config=None, *, penalized=True, verbose=False
):
    """Fit separate global previous- and within-round conformity terms."""
    return _fit_comparison_model(
        matrices,
        SPLIT_GLOBAL_CONFORMITY_SPEC,
        config,
        penalized=penalized,
        verbose=verbose,
    )


def fit_paper_comparison_model(
    matrices, config=None, *, penalized=True, verbose=False
):
    """Fit the paper specification with model-specific conformity terms."""
    return _fit_comparison_model(
        matrices,
        PAPER_COMPARISON_SPEC,
        config,
        penalized=penalized,
        verbose=verbose,
    )


def fit_interaction_model(matrices, config=None, *, penalized=True, verbose=False):
    """Fit the paper model plus model-specific inertia-conformity interactions."""
    return _fit_comparison_model(
        matrices, INTERACTION_SPEC, config, penalized=penalized, verbose=verbose
    )


def fit_debate_random_intercept_model(
    matrices, config=None, *, penalized=True, verbose=False
):
    """Fit the paper model plus penalized debate-by-verdict random effects.

    This is a MAP sensitivity analysis with a fixed Gaussian prior scale, not a
    marginal-likelihood mixed model. It is consequently excluded from the MLE
    AIC ranking produced by ``scripts/models/fit_models.py``.
    """
    return _fit_comparison_model(
        matrices,
        DEBATE_RANDOM_INTERCEPT_SPEC,
        config,
        penalized=penalized,
        verbose=verbose,
    )


def model_coefficients(fit: ModelFit) -> pd.DataFrame:
    """Return interpretable behavioral coefficients from any comparison fit."""
    if not isinstance(fit.model, ComparisonModel):
        raise TypeError("model_coefficients requires a fitted ComparisonModel")
    rows = []
    aliases = {
        "gamma_prev_m": "gamma_prev",
        "gamma_within_m": "gamma_within",
        "delta_prev_m": "delta_prev",
        "delta_within_m": "delta_within",
    }
    for name, parameter in fit.model.named_parameters():
        if name in {"theta_raw", "phi_raw", "debate_raw"}:
            continue
        values = parameter.detach().cpu().reshape(-1).numpy()
        labels = MODEL_NAMES[: len(values)] if len(values) > 1 else ("All models",)
        for label, estimate in zip(labels, values):
            rows.append(
                {
                    "fit_model": fit.name,
                    "model": label,
                    "parameter": aliases.get(name, name),
                    "estimate": float(estimate),
                    "odds_ratio": float(np.exp(estimate)),
                }
            )
    if hasattr(fit.model, "debate_raw"):
        centered = fit.model.debate_raw - fit.model.debate_raw.mean(dim=1, keepdim=True)
        rows.append(
            {
                "fit_model": fit.name,
                "model": "All debates",
                "parameter": "debate_effect_sd",
                "estimate": float(centered.detach().cpu().std()),
                "odds_ratio": np.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "fit_model",
            "model",
            "parameter",
            "estimate",
            "odds_ratio",
        ),
    )


def fit_paper_model(
    matrices: ModelMatrices,
    config: PaperModelConfig | None = None,
    *,
    verbose: bool = True,
) -> ModelFit:
    """Fit the historical paper model with minibatch AdamW.

    The loss is mean multinomial negative log-likelihood plus explicit Gaussian
    prior penalties divided by the number of observations. This reproduces the
    historical fitting path, including AdamW's configured decoupled weight
    decay. The best full-data objective is retained and restored after early
    stopping.

    Parameters
    ----------
    matrices
        Preprocessed observations and predictor matrices.
    config
        Optimization settings and prior scales. Defaults reproduce the paper
        notebook: 120 epochs, learning rate 0.02, and seed 0.
    verbose
        Print the full-data objective after each epoch.
    """
    config = config or PaperModelConfig()
    torch.manual_seed(config.seed)
    device = config.device
    y = torch.as_tensor(matrices.y, dtype=torch.long, device=device)
    model_idx = torch.as_tensor(matrices.model_idx, dtype=torch.long, device=device)
    dilemma_idx = torch.as_tensor(matrices.dilemma_idx, dtype=torch.long, device=device)
    same_prev = torch.as_tensor(matrices.same_prev_mat, dtype=torch.float32, device=device)
    exposure_prev = torch.as_tensor(
        matrices.exposure_prev_mat, dtype=torch.float32, device=device
    )
    exposure_within = torch.as_tensor(
        matrices.exposure_within_mat, dtype=torch.float32, device=device
    )

    model = PaperModel(
        matrices.n_models, matrices.n_dilemmas, matrices.n_verdicts
    ).to(device)
    criterion = nn.NLLLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(
            model_idx,
            dilemma_idx,
            same_prev,
            exposure_prev,
            exposure_within,
            y,
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )

    def prior_term():
        """Return the negative log-prior on the mean-loss scale."""
        return (
            model.theta_raw.pow(2).sum() / (2 * config.sigma_theta**2)
            + model.phi_raw.pow(2).sum() / (2 * config.sigma_phi**2)
            + model.alpha.pow(2).sum() / (2 * config.sigma_alpha**2)
            + model.gamma_prev_m.pow(2).sum() / (2 * config.sigma_gamma_prev**2)
            + model.gamma_within_m.pow(2).sum()
            / (2 * config.sigma_gamma_within**2)
        ) / matrices.n_observations

    best_objective = float("inf")
    best_state = None
    no_improvement = 0
    nll_value = float("nan")
    objective = float("nan")

    for epoch in range(config.epochs):
        model.train()
        for mi, di, sp, ep, ew, target in loader:
            # The explicit prior is evaluated for every minibatch but divided by
            # global N, matching the original notebook implementation.
            loss = criterion(model(mi, di, sp, ep, ew), target) + prior_term()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            nll_value = criterion(
                model(model_idx, dilemma_idx, same_prev, exposure_prev, exposure_within),
                y,
            ).item()
            objective = nll_value + float(prior_term())

        if verbose:
            print(
                f"paper epoch {epoch + 1:3d}: objective={objective:.6f} "
                f"nll={nll_value:.6f}"
            )
        # A tolerance avoids treating tiny floating-point changes as meaningful
        # improvement. Saving CPU copies prevents later optimizer steps from
        # mutating the retained state.
        if objective + config.tolerance < best_objective:
            best_objective = objective
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        nll_value = criterion(
            model(model_idx, dilemma_idx, same_prev, exposure_prev, exposure_within), y
        ).item()
        objective = nll_value + float(prior_term())

    return ModelFit(
        name=MAIN_MODEL_NAME,
        model=model,
        config=config,
        n_observations=matrices.n_observations,
        n_parameters=(matrices.n_models + matrices.n_dilemmas)
        * (matrices.n_verdicts - 1)
        + 3 * matrices.n_models,
        epochs_fit=epoch + 1,
        negative_log_likelihood=nll_value,
        objective=objective,
        penalized=True,
    )


def fit_paper_model_lbfgs(
    matrices: ModelMatrices,
    config: PaperModelConfig | None = None,
) -> ModelFit:
    """Fit the explicit paper-model MAP objective with full-batch L-BFGS.

    Conditional on the precomputed predictors, this penalized multinomial model
    has a convex objective. L-BFGS reaches that objective far faster than the
    historical minibatch optimizer, which makes 200 full bootstrap refits
    practical. This path includes the explicit Normal priors but intentionally
    does not emulate AdamW's optimizer-specific decoupled weight decay.
    """
    config = config or PaperModelConfig()
    torch.manual_seed(config.seed)
    device = config.device
    y = torch.as_tensor(matrices.y, dtype=torch.long, device=device)
    model_idx = torch.as_tensor(matrices.model_idx, dtype=torch.long, device=device)
    dilemma_idx = torch.as_tensor(matrices.dilemma_idx, dtype=torch.long, device=device)
    same_prev = torch.as_tensor(matrices.same_prev_mat, dtype=torch.float32, device=device)
    exposure_prev = torch.as_tensor(
        matrices.exposure_prev_mat, dtype=torch.float32, device=device
    )
    exposure_within = torch.as_tensor(
        matrices.exposure_within_mat, dtype=torch.float32, device=device
    )
    model = PaperModel(
        matrices.n_models, matrices.n_dilemmas, matrices.n_verdicts
    ).to(device)
    criterion = nn.NLLLoss(reduction="mean")
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=config.epochs,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def prior_term():
        """Return the explicit Gaussian-prior penalty on the mean-loss scale."""
        return (
            model.theta_raw.pow(2).sum() / (2 * config.sigma_theta**2)
            + model.phi_raw.pow(2).sum() / (2 * config.sigma_phi**2)
            + model.alpha.pow(2).sum() / (2 * config.sigma_alpha**2)
            + model.gamma_prev_m.pow(2).sum() / (2 * config.sigma_gamma_prev**2)
            + model.gamma_within_m.pow(2).sum()
            / (2 * config.sigma_gamma_within**2)
        ) / matrices.n_observations

    def closure():
        """Recompute loss and gradients as required by L-BFGS line search."""
        optimizer.zero_grad()
        loss = criterion(
            model(model_idx, dilemma_idx, same_prev, exposure_prev, exposure_within), y
        ) + prior_term()
        loss.backward()
        return loss

    optimizer.step(closure)
    model.eval()
    with torch.no_grad():
        negative_log_likelihood = criterion(
            model(model_idx, dilemma_idx, same_prev, exposure_prev, exposure_within), y
        ).item()
        objective = negative_log_likelihood + float(prior_term())
    state = optimizer.state.get(model.theta_raw, {})
    iterations = int(state.get("n_iter", config.epochs))
    return ModelFit(
        name=MAIN_MODEL_NAME,
        model=model,
        config=config,
        n_observations=matrices.n_observations,
        n_parameters=(matrices.n_models + matrices.n_dilemmas)
        * (matrices.n_verdicts - 1)
        + 3 * matrices.n_models,
        epochs_fit=iterations,
        negative_log_likelihood=negative_log_likelihood,
        objective=objective,
        penalized=True,
    )


def paper_model_coefficients(fit: ModelFit) -> pd.DataFrame:
    """Return behavioral coefficients and their per-unit odds ratios.

    Fixed effects are omitted because Table 1 reports only inertia and the two
    conformity terms. Exponentiating a coefficient gives the multiplicative
    change in odds for a one-unit increase in its predictor.
    """
    if not isinstance(fit.model, PaperModel):
        raise TypeError("paper_model_coefficients requires a fitted PaperModel")
    rows = []
    parameter_values = {
        "alpha": fit.model.alpha,
        "gamma_prev": fit.model.gamma_prev_m,
        "gamma_within": fit.model.gamma_within_m,
    }
    for parameter, values in parameter_values.items():
        for model_name, estimate in zip(MODEL_NAMES, values.detach().cpu().numpy()):
            rows.append(
                {
                    "model": model_name,
                    "parameter": parameter,
                    "estimate": float(estimate),
                    "odds_ratio": float(np.exp(estimate)),
                }
            )
    return pd.DataFrame(rows)


def resample_dilemmas(
    matrices: ModelMatrices,
    sampled_dilemmas: np.ndarray,
) -> ModelMatrices:
    """Construct one nonparametric dilemma-cluster bootstrap sample.

    ``sampled_dilemmas`` contains one dilemma ID per draw and may contain the
    same ID multiple times. If dilemma 3 is drawn twice, every observation for
    dilemma 3 is repeated twice. This multiplicity is the defining difference
    from the former ``np.isin`` implementation, which reduced the operation to
    subsampling the unique drawn IDs.

    All experiment-specific debates for a dilemma move together, preserving the
    intended clustering by underlying AITA scenario.
    """
    if sampled_dilemmas.ndim != 1:
        raise ValueError("sampled_dilemmas must be one-dimensional")
    # Convert the draw into an integer weight for each dilemma. Indexing those
    # weights by each row's dilemma ID gives an efficient O(N) implementation;
    # it avoids scanning all rows once per sampled cluster.
    counts = np.bincount(
        sampled_dilemmas.astype(np.int64), minlength=matrices.n_dilemmas
    )
    row_repetitions = counts[matrices.dilemma_idx]

    def repeat(values):
        """Apply the same cluster multiplicity to one aligned matrix."""
        return np.repeat(values, row_repetitions, axis=0)

    return ModelMatrices(
        y=repeat(matrices.y),
        model_idx=repeat(matrices.model_idx),
        dilemma_idx=repeat(matrices.dilemma_idx),
        round_idx=repeat(matrices.round_idx),
        speaker_pos=repeat(matrices.speaker_pos),
        same_prev_mat=repeat(matrices.same_prev_mat),
        exposure_prev_mat=repeat(matrices.exposure_prev_mat),
        exposure_within_mat=repeat(matrices.exposure_within_mat),
        is_sync=repeat(matrices.is_sync),
        debate_idx=repeat(matrices.debate_idx),
    )


_BOOTSTRAP_MATRICES: ModelMatrices | None = None
_BOOTSTRAP_CONFIG: PaperModelConfig | None = None


def _initialize_bootstrap_worker(
    matrices: ModelMatrices,
    config: PaperModelConfig,
) -> None:
    """Load shared inputs once per process and prevent CPU oversubscription.

    Passing the full matrix bundle with every replication would repeatedly
    serialize tens of thousands of rows. A pool initializer sends it once to
    each worker. Because the outer pool already parallelizes across CPU cores,
    each torch worker is restricted to one internal compute thread.
    """
    global _BOOTSTRAP_MATRICES, _BOOTSTRAP_CONFIG
    _BOOTSTRAP_MATRICES = matrices
    _BOOTSTRAP_CONFIG = config
    torch.set_num_threads(1)


def _fit_bootstrap_replication(
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Draw clusters, refit the paper model, and return behavioral coefficients."""
    if _BOOTSTRAP_MATRICES is None or _BOOTSTRAP_CONFIG is None:
        raise RuntimeError("Bootstrap worker was not initialized")
    # The seed controls both the cluster draw and the fit, making every
    # replication reproducible independently of multiprocessing completion order.
    rng = np.random.default_rng(seed)
    dilemmas = np.arange(_BOOTSTRAP_MATRICES.n_dilemmas)
    sampled_dilemmas = rng.choice(dilemmas, size=len(dilemmas), replace=True)
    bootstrap_matrices = resample_dilemmas(_BOOTSTRAP_MATRICES, sampled_dilemmas)
    fit = fit_paper_model_lbfgs(
        bootstrap_matrices,
        replace(_BOOTSTRAP_CONFIG, seed=seed),
    )
    model = fit.model
    if not isinstance(model, PaperModel):
        raise TypeError("Expected fit_paper_model to return a PaperModel")
    return (
        seed,
        model.alpha.detach().cpu().numpy().copy(),
        model.gamma_prev_m.detach().cpu().numpy().copy(),
        model.gamma_within_m.detach().cpu().numpy().copy(),
    )


def bootstrap_paper_model(
    matrices: ModelMatrices,
    config: PaperModelConfig | None = None,
    *,
    n_bootstrap: int = 200,
    n_jobs: int | None = None,
    base_seed: int = 2332,
) -> BootstrapResult:
    """Estimate uncertainty with a proper dilemma-cluster bootstrap.

    Each replication draws ``n_dilemmas`` IDs with replacement, repeats every
    selected dilemma according to its draw count, and refits the explicit MAP
    objective. Results are returned in deterministic seed order even though
    workers complete asynchronously.

    Parameters
    ----------
    matrices
        Full preprocessed data from which clusters are drawn.
    config
        Prior scales, device, and maximum L-BFGS iterations.
    n_bootstrap
        Number of independent cluster samples.
    n_jobs
        Number of worker processes. Defaults to one fewer than the detected CPU
        count. Each worker uses one torch thread.
    base_seed
        Seed of the first replication; subsequent fits use consecutive seeds.

    Returns
    -------
    BootstrapResult
        Model-specific draws for inertia, previous-round conformity, and
        within-round conformity.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    config = config or PaperModelConfig()
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    seeds = [base_seed + index for index in range(n_bootstrap)]
    # ``spawn`` avoids inheriting torch thread-pool state from the parent process
    # and works consistently across macOS and Linux.
    context = mp.get_context("spawn")
    with context.Pool(
        processes=n_jobs,
        initializer=_initialize_bootstrap_worker,
        initargs=(matrices, config),
    ) as pool:
        replications = []
        for completed, replication in enumerate(
            pool.imap_unordered(_fit_bootstrap_replication, seeds, chunksize=1),
            start=1,
        ):
            replications.append(replication)
            if completed % 10 == 0 or completed == n_bootstrap:
                print(f"Completed {completed}/{n_bootstrap} bootstrap fits.", flush=True)
    # ``imap_unordered`` keeps every worker busy; sorting restores seed order for
    # reproducible saved arrays and easier replication-level debugging.
    replications.sort(key=lambda replication: replication[0])
    return BootstrapResult(
        alpha=np.stack([replication[1] for replication in replications]),
        gamma_prev=np.stack([replication[2] for replication in replications]),
        gamma_within=np.stack([replication[3] for replication in replications]),
    )


ModelFitter = Callable[[ModelMatrices, FitConfig], ModelFit]
MODEL_FITTERS: dict[str, Callable[..., ModelFit]] = {
    MAIN_MODEL_NAME: fit_paper_model,
}

# Models used for the nested AIC comparison. Each supports both an explicitly
# penalized MAP fit and an unpenalized MLE refit on the same matrices.
COMPARISON_MODEL_FITTERS: dict[str, Callable[..., ModelFit]] = {
    FIXED_EFFECTS_SPEC.name: fit_fixed_effects_model,
    INERTIA_SPEC.name: fit_inertia_model,
    GLOBAL_CONFORMITY_SPEC.name: fit_global_conformity_model,
    PREVIOUS_CONFORMITY_SPEC.name: fit_previous_round_conformity_model,
    SPLIT_GLOBAL_CONFORMITY_SPEC.name: fit_split_global_conformity_model,
    PAPER_COMPARISON_SPEC.name: fit_paper_comparison_model,
    INTERACTION_SPEC.name: fit_interaction_model,
}

SENSITIVITY_MODEL_FITTERS: dict[str, Callable[..., ModelFit]] = {
    DEBATE_RANDOM_INTERCEPT_SPEC.name: fit_debate_random_intercept_model,
}


# Compatibility names for the original notebook and review-response scripts.
K = N_VERDICTS
model_idxs = MODEL_INDICES
DeliberationModelSplitPerModel = PaperModel


def fit_map_split_per_model(exp, **kwargs):
    """Fit the paper model through the historical dictionary-based interface.

    This wrapper translates legacy keyword names and returns only the torch
    module, matching the contract used by ``notebooks/fit_model.ipynb``.
    New code should call :func:`fit_paper_model` with :class:`ModelMatrices`.
    """
    matrices = ModelMatrices.from_legacy_dict(exp)
    config = PaperModelConfig(
        learning_rate=kwargs.pop("lr", 1e-2),
        epochs=kwargs.pop("epochs", 60),
        batch_size=kwargs.pop("batch_size", 4096),
        seed=kwargs.pop("seed", 0),
        device=kwargs.pop("device", None)
        or ("cuda" if torch.cuda.is_available() else "cpu"),
        tolerance=kwargs.pop("tol", 1e-4),
        early_stopping_patience=kwargs.pop("early_stop_patience", 10),
        sigma_theta=kwargs.pop("sigma_theta", 1.0),
        sigma_phi=kwargs.pop("sigma_phi", 1.0),
        sigma_alpha=kwargs.pop("sigma_alpha", 0.5),
        sigma_gamma_prev=kwargs.pop("sigma_gamma_prev", 0.5),
        sigma_gamma_within=kwargs.pop("sigma_gamma_within", 0.5),
    )
    verbose = kwargs.pop("verbose", False)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected fitting arguments: {unknown}")
    return fit_paper_model(matrices, config, verbose=verbose).model
