"""Maximum-likelihood Bayesian Knowledge Tracing by Baum-Welch EM.

The E-step smooths every learner's independent per-skill mastery chains and
accumulates expected state and transition counts. The M-step sets each skill's
parameters to their expected-count ratios, which maximize the expected
complete-data log-likelihood in closed form. Slip and guess are clipped to
their bounds; each has a concave Bernoulli objective, so the clipped ratio is
the exact constrained maximizer and no iteration decreases the observed-data
log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtValidationError
from mirt.models.dynamic import BKTModel, BKTResult

_PARAMETERS = ("p_init", "p_learn", "p_forget", "p_slip", "p_guess")


@dataclass
class BKTEMResult(BKTResult):
    """Result from maximum-likelihood BKT estimation by Baum-Welch EM.

    Attributes
    ----------
    n_iterations : int
        EM iterations (M-steps) taken by the selected start.
    log_likelihood_history : NDArray
        Observed-data log-likelihood of the selected start at its starting
        values and after every M-step. It never decreases.
    start_log_likelihoods : NDArray
        Final log-likelihood reached from every start, in start order. The
        reported estimates come from the largest.
    """

    n_iterations: int
    log_likelihood_history: NDArray[np.float64]
    start_log_likelihoods: NDArray[np.float64]

    def summary(self) -> str:
        """Return the estimation summary with the EM iteration and start counts."""
        lines = super().summary().split("\n")
        position = next(
            (
                index + 1
                for index, line in enumerate(lines)
                if line.startswith("Converged:")
            ),
            len(lines) - 1,
        )
        lines[position:position] = [
            f"Iterations:         {self.n_iterations}",
            f"Starts:             {self.start_log_likelihoods.size}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class _ChainLayout:
    """Same-skill opportunity links of a validated trial layout.

    Arrays share the shape of ``skills``: ``(n_trials,)`` for a shared layout
    or ``(n_persons, n_trials)`` for person-specific layouts. ``previous``
    holds each trial's previous opportunity for the same skill, or the trial
    itself at a chain's first opportunity.
    """

    skills: NDArray[np.int_]
    previous: NDArray[np.intp]
    first: NDArray[np.bool_]
    last: NDArray[np.bool_]
    n_skills: int

    @classmethod
    def from_skills(cls, skills: NDArray[np.int_], n_skills: int) -> _ChainLayout:
        """Link every trial to its skill's adjacent opportunities."""
        rows = np.atleast_2d(skills)
        order = np.argsort(rows, axis=1, kind="stable")
        ordered = np.take_along_axis(rows, order, axis=1)
        same = ordered[:, 1:] == ordered[:, :-1]
        previous = np.empty_like(order)
        following = np.empty_like(order)
        np.put_along_axis(
            previous,
            order,
            np.concatenate(
                [order[:, :1], np.where(same, order[:, :-1], order[:, 1:])], axis=1
            ),
            axis=1,
        )
        np.put_along_axis(
            following,
            order,
            np.concatenate(
                [np.where(same, order[:, 1:], order[:, :-1]), order[:, -1:]], axis=1
            ),
            axis=1,
        )
        trials = np.arange(rows.shape[1])
        previous = previous.reshape(skills.shape)
        return cls(
            skills=skills,
            previous=previous,
            first=previous == trials,
            last=following.reshape(skills.shape) == trials,
            n_skills=n_skills,
        )

    def per_skill(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sum ``(n_persons, n_trials)`` values over each skill's trials."""
        if self.skills.ndim == 1:
            return np.bincount(
                self.skills, weights=values.sum(axis=0), minlength=self.n_skills
            )
        return np.bincount(
            self.skills.ravel(), weights=values.ravel(), minlength=self.n_skills
        )

    def at_previous(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return each trial's value at its chain's previous opportunity."""
        if self.skills.ndim == 1:
            return values[:, self.previous]
        return np.take_along_axis(values, self.previous, axis=1)


@dataclass(frozen=True)
class _ResponseCounts:
    """Fixed response indicators and chain counts reused by every E-step."""

    observed: NDArray[np.bool_]
    correct: NDArray[np.bool_]
    incorrect: NDArray[np.bool_]
    n_chains: NDArray[np.float64]

    @classmethod
    def from_responses(
        cls, responses: NDArray[np.int_], layout: _ChainLayout
    ) -> _ResponseCounts:
        first = np.broadcast_to(layout.first, responses.shape)
        return cls(
            observed=responses >= 0,
            correct=responses == 1,
            incorrect=responses == 0,
            n_chains=layout.per_skill(first.astype(np.float64)),
        )


def _prepare_bkt_fit(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    n_skills: int | None,
    allow_forgetting: bool,
    *,
    use_rust: bool,
    person_layouts: bool,
    skill_names: list[str] | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_], BKTModel]:
    """Validate BKT fitting data and construct the matching model.

    ``person_layouts`` additionally accepts a skill matrix matching
    ``responses``; otherwise the layout must be shared by every learner.
    """
    responses = np.asarray(responses)
    skill_assignments = np.asarray(skill_assignments)

    if responses.ndim != 2:
        raise MirtValidationError("responses must have shape (n_persons, n_trials)")
    if responses.shape[0] == 0:
        raise MirtValidationError("responses must contain at least one person")
    if responses.shape[1] == 0:
        raise MirtValidationError("responses must contain at least one trial")
    if not person_layouts and skill_assignments.ndim != 1:
        raise MirtValidationError("skill_assignments must be one-dimensional")
    if skill_assignments.ndim == 1 and len(skill_assignments) != responses.shape[1]:
        raise MirtValidationError(
            "skill_assignments length must match the number of trials"
        )
    if skill_assignments.ndim != 1 and skill_assignments.shape != responses.shape:
        raise MirtValidationError(
            "skill_assignments must be one-dimensional or match responses"
        )
    if not np.issubdtype(skill_assignments.dtype, np.integer):
        raise MirtValidationError("skill_assignments must contain integer values")
    if np.any(skill_assignments < 0):
        raise MirtValidationError("skill_assignments must contain non-negative values")
    if not isinstance(allow_forgetting, (bool, np.bool_)):
        raise TypeError("allow_forgetting must be a boolean")

    if n_skills is None:
        n_skills = int(np.max(skill_assignments)) + 1
    model = BKTModel(
        n_skills=n_skills,
        skill_names=skill_names,
        allow_forgetting=bool(allow_forgetting),
        use_rust=use_rust,
    )
    responses, skill_assignments = model._validate_batch(responses, skill_assignments)
    if not np.any(responses >= 0):
        raise MirtValidationError("responses must contain at least one observed value")
    return responses, skill_assignments, model


def _bkt_result_fields(
    model: BKTModel,
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    *,
    n_fixed_per_skill: int = 0,
) -> dict[str, Any]:
    """Return the :class:`BKTResult` fields shared by every BKT estimator.

    Learning curves average a learner's smoothed mastery over their
    opportunities for a skill, and skill mastery is the smoothed mastery at
    the final opportunity. Both are zero for skills a learner never
    practiced. AIC and BIC count four parameters per skill, or five when
    forgetting is estimated, less ``n_fixed_per_skill`` parameters held fixed
    for every skill.
    """
    gamma, log_likelihoods = model.forward_backward_batch(responses, skill_assignments)
    learned = gamma[..., 1]
    n_persons = responses.shape[0]
    n_skills = model.n_skills
    n_cells = n_persons * n_skills
    cells = (
        np.arange(n_persons)[:, None] * n_skills
        + np.broadcast_to(skill_assignments, responses.shape)
    ).ravel()
    totals = np.bincount(cells, weights=learned.ravel(), minlength=n_cells)
    counts = np.bincount(cells, minlength=n_cells)
    learning_curves = np.divide(
        totals, counts, out=np.zeros(n_cells), where=counts > 0
    ).reshape(n_persons, n_skills)

    last = np.broadcast_to(
        _ChainLayout.from_skills(skill_assignments, n_skills).last,
        responses.shape,
    )
    skill_mastery = np.zeros(n_cells)
    skill_mastery[cells[last.ravel()]] = learned[last]

    log_likelihood = float(log_likelihoods.sum())
    n_parameters = ((5 if model.allow_forgetting else 4) - n_fixed_per_skill) * n_skills
    n_observations = int(np.count_nonzero(responses >= 0))
    return {
        "model": model,
        "learning_curves": learning_curves,
        "skill_mastery": skill_mastery.reshape(n_persons, n_skills),
        "log_likelihood": log_likelihood,
        "aic": -2.0 * log_likelihood + 2.0 * n_parameters,
        "bic": -2.0 * log_likelihood + float(np.log(n_observations)) * n_parameters,
        "n_observations": n_observations,
        "n_parameters": n_parameters,
    }


def _ratio(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    current: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return expected-count ratios, keeping parameters the data never reach."""
    ratio = np.divide(numerator, denominator, out=current.copy(), where=denominator > 0)
    return np.clip(ratio, 0.0, 1.0)


def _em_update(
    model: BKTModel,
    responses: NDArray[np.int_],
    layout: _ChainLayout,
    counts: _ResponseCounts,
    slip_bounds: tuple[float, float],
    guess_bounds: tuple[float, float],
) -> tuple[float, dict[str, NDArray[np.float64]]]:
    """Return the current log-likelihood and the closed-form M-step update."""
    transitions = None
    if model.allow_forgetting:
        transitions = np.empty((2, responses.shape[1], responses.shape[0]))
        gamma, log_likelihoods = model._forward_backward_layouts(
            responses, layout.skills, transition_out=transitions
        )
    else:
        gamma, log_likelihoods = model._forward_backward_batch_validated(
            responses, layout.skills
        )
    unlearned = gamma[..., 0]
    learned = gamma[..., 1]
    per_skill = layout.per_skill
    initial = per_skill(learned * layout.first)
    continuing = ~layout.last
    update = {
        "p_init": _ratio(initial, counts.n_chains, model.p_init),
        "p_slip": np.clip(
            _ratio(
                per_skill(learned * counts.incorrect),
                per_skill(learned * counts.observed),
                model.p_slip,
            ),
            *slip_bounds,
        ),
        "p_guess": np.clip(
            _ratio(
                per_skill(unlearned * counts.correct),
                per_skill(unlearned * counts.observed),
                model.p_guess,
            ),
            *guess_bounds,
        ),
    }

    if transitions is None:
        # Learning is the only way into the absorbing learned state, so a
        # chain's expected learning transitions telescope to its final minus
        # its initial mastery.
        learned_transitions = per_skill(learned * layout.last) - initial
    else:
        later = ~layout.first
        learned_transitions = per_skill(
            layout.at_previous(unlearned) * transitions[0].T * later
        )
        update["p_forget"] = _ratio(
            per_skill(layout.at_previous(learned) * transitions[1].T * later),
            per_skill(learned * continuing),
            model.p_forget,
        )
    update["p_learn"] = _ratio(
        learned_transitions, per_skill(unlearned * continuing), model.p_learn
    )
    return float(log_likelihoods.sum()), update


def _run_em(
    model: BKTModel,
    responses: NDArray[np.int_],
    layout: _ChainLayout,
    counts: _ResponseCounts,
    *,
    max_iter: int,
    tol: float,
    slip_bounds: tuple[float, float],
    guess_bounds: tuple[float, float],
) -> tuple[NDArray[np.float64], bool]:
    """Iterate EM in place; return the log-likelihood trace and convergence.

    The model keeps the parameters at which the final log-likelihood was
    evaluated, so the trace ends at the returned estimates.
    """
    history: list[float] = []
    for iteration in range(max_iter + 1):
        log_likelihood, update = _em_update(
            model, responses, layout, counts, slip_bounds, guess_bounds
        )
        history.append(log_likelihood)
        if iteration > 0 and abs(log_likelihood - history[-2]) < tol:
            return np.asarray(history), True
        if iteration == max_iter:
            break
        for name, values in update.items():
            setattr(model, name, values)
    return np.asarray(history), False


def _start_model(
    template: BKTModel,
    start: int,
    rng: np.random.Generator,
    slip_bounds: tuple[float, float],
    guess_bounds: tuple[float, float],
) -> BKTModel:
    """Return the model holding one start's parameter values.

    The first start takes the template's values, which are the
    :class:`BKTModel` defaults; later starts are random.
    """
    n_skills = template.n_skills
    if start == 0:
        values = {name: getattr(template, name) for name in _PARAMETERS}
    else:
        values = {
            "p_init": rng.uniform(0.05, 0.95, n_skills),
            "p_learn": rng.uniform(0.01, 0.5, n_skills),
            "p_forget": rng.uniform(0.001, 0.1, n_skills),
        }
        for name, (lower, upper) in (
            ("p_slip", slip_bounds),
            ("p_guess", guess_bounds),
        ):
            values[name] = rng.uniform(lower, min(upper, max(lower, 0.4)), n_skills)
    if not template.allow_forgetting:
        values["p_forget"] = np.zeros(n_skills)
    values["p_slip"] = np.clip(values["p_slip"], *slip_bounds)
    values["p_guess"] = np.clip(values["p_guess"], *guess_bounds)
    return BKTModel(
        n_skills=n_skills,
        skill_names=template.skill_names,
        allow_forgetting=template.allow_forgetting,
        use_rust=template.use_rust,
        **values,
    )


def _validated_count(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise MirtValidationError(f"{name} must be a positive integer")
    return int(value)


def _validated_bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    try:
        values = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            f"{name} must be (lower, upper) with 0 < lower <= upper < 1"
        ) from exc
    if (
        values.shape != (2,)
        or not np.all(np.isfinite(values))
        or not 0.0 < values[0] <= values[1] < 1.0
    ):
        raise MirtValidationError(
            f"{name} must be (lower, upper) with 0 < lower <= upper < 1"
        )
    return float(values[0]), float(values[1])


def fit_bkt_em(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    *,
    n_skills: int | None = None,
    skill_names: list[str] | None = None,
    allow_forgetting: bool = False,
    max_iter: int = 500,
    tol: float = 1e-4,
    n_starts: int = 1,
    seed: int | None = None,
    slip_bounds: tuple[float, float] = (1e-4, 0.5),
    guess_bounds: tuple[float, float] = (1e-4, 0.5),
    use_rust: bool = True,
) -> BKTEMResult:
    """Fit Bayesian Knowledge Tracing by maximum likelihood (Baum-Welch EM).

    Each skill has its own initial mastery, learning, optional forgetting,
    slip and guess probabilities. Every EM iteration smooths all learners'
    per-skill mastery chains together and updates each parameter to its
    closed-form expected-count ratio, so the log-likelihood never decreases.

    Parameters
    ----------
    responses : NDArray
        Integer response matrix with shape ``(n_persons, n_trials)``: ``1``
        correct, ``0`` incorrect and ``-1`` missing. A missing response adds
        no evidence but keeps its place in the skill's opportunity sequence.
    skill_assignments : NDArray
        Skill index of every trial, either a shared ``(n_trials,)`` layout or
        a person-specific matrix matching ``responses``.
    n_skills : int, optional
        Number of skills. Inferred from ``skill_assignments`` if omitted.
    skill_names : list of str, optional
        Names for the fitted model's skills.
    allow_forgetting : bool, default=False
        Whether to estimate a per-skill forgetting probability.
    max_iter : int, default=500
        Maximum number of EM iterations per start.
    tol : float, default=1e-4
        Convergence tolerance for the absolute change in the total
        log-likelihood between iterations, as in
        :class:`~mirt.estimation.em.EMEstimator`.
    n_starts : int, default=1
        Number of starts. The first uses fixed default values and later
        starts draw random values; the start with the largest final
        log-likelihood is returned.
    seed : int, optional
        Seed for the random starts.
    slip_bounds, guess_bounds : tuple of float, default=(1e-4, 0.5)
        Closed bounds ``(lower, upper)`` with ``0 < lower <= upper < 1``.
        Upper bounds of at most 0.5 keep a correct response at least as
        likely in the learned state as in the unlearned state, which rules
        out label-swapped solutions. Equal bounds fix the parameter.
    use_rust : bool, default=True
        Use compiled smoothing kernels when available. Estimation with
        forgetting always uses the vectorized NumPy smoother, which also
        returns the transition posteriors that model needs.

    Returns
    -------
    BKTEMResult
        Fitted model, learner summaries, information criteria and EM
        diagnostics. ``converged`` is ``False`` when the selected start used
        all ``max_iter`` iterations without meeting ``tol``.

    Notes
    -----
    Skills or parameters that receive no expected data (for example the
    learning rate of a skill practiced once per learner) keep their starting
    values. AIC and BIC count four parameters per skill, or five with
    forgetting, as :class:`~mirt.estimation.dynamic_gibbs.BKTGibbsSampler`
    does, less any slip or guess probability fixed by equal bounds.

    Examples
    --------
    >>> import numpy as np
    >>> from mirt.models import BKTModel
    >>> generator = BKTModel(n_skills=2, p_learn=np.array([0.15, 0.25]))
    >>> responses, skills, _ = generator.simulate(500, 10, seed=1)
    >>> result = fit_bkt_em(responses, skills)
    >>> result.converged
    True
    """
    max_iter = _validated_count(max_iter, "max_iter")
    n_starts = _validated_count(n_starts, "n_starts")
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, Real)
        or not np.isfinite(tol)
        or tol <= 0.0
    ):
        raise MirtValidationError("tol must be a positive finite number")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise MirtValidationError("seed must be a non-negative integer or None")
    slip_bounds = _validated_bounds(slip_bounds, "slip_bounds")
    guess_bounds = _validated_bounds(guess_bounds, "guess_bounds")
    if not isinstance(use_rust, (bool, np.bool_)):
        raise TypeError("use_rust must be a boolean")

    responses, skill_assignments, template = _prepare_bkt_fit(
        responses,
        skill_assignments,
        n_skills,
        allow_forgetting,
        use_rust=bool(use_rust),
        person_layouts=True,
        skill_names=skill_names,
    )
    if skill_assignments.ndim == 2 and np.all(
        skill_assignments == skill_assignments[0]
    ):
        skill_assignments = skill_assignments[0]
    layout = _ChainLayout.from_skills(skill_assignments, template.n_skills)
    counts = _ResponseCounts.from_responses(responses, layout)
    rng = np.random.default_rng(seed)

    fits: list[tuple[BKTModel, NDArray[np.float64], bool]] = []
    for start in range(n_starts):
        model = _start_model(template, start, rng, slip_bounds, guess_bounds)
        history, converged = _run_em(
            model,
            responses,
            layout,
            counts,
            max_iter=max_iter,
            tol=float(tol),
            slip_bounds=slip_bounds,
            guess_bounds=guess_bounds,
        )
        fits.append((model, history, converged))

    start_log_likelihoods = np.array([trace[-1] for _, trace, _ in fits])
    model, history, converged = fits[int(np.argmax(start_log_likelihoods))]
    n_fixed = sum(lower == upper for lower, upper in (slip_bounds, guess_bounds))
    return BKTEMResult(
        **_bkt_result_fields(
            model, responses, skill_assignments, n_fixed_per_skill=n_fixed
        ),
        converged=converged,
        n_iterations=history.size - 1,
        log_likelihood_history=history,
        start_log_likelihoods=start_log_likelihoods,
    )
