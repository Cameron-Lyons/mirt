from abc import abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._logistic import (
    _logistic_probability,
    _scaled_information,
    _sigmoid_derivative,
)
from mirt.exceptions import MirtValidationError
from mirt.models.base import DichotomousItemModel

_MAX_DOUBLE_EXP_INPUT = 50.0
_FIVE_PL_CURVE_CHUNK_ELEMENTS = 262_144
_UNIDIMENSIONAL_CURVE_CHUNK_ELEMENTS = 262_144
_LOGISTIC_CURVE_CHUNK_ELEMENTS = 262_144


def _bounded_exponential(value: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponentiate safely for links containing a second exponential."""
    result = np.minimum(value, _MAX_DOUBLE_EXP_INPUT)
    with np.errstate(under="ignore"):
        np.exp(result, out=result)
    return result


def _double_exponential_log_information(
    logits: NDArray[np.float64], discrimination: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Recover information when the unscaled double-exponential tail underflows."""
    # I = a² exp(2z - exp(z)) / (1 - exp(-exp(z))). In the left tail
    # its log is 2 log|a| + z to floating-point precision.
    result = 2.0 * np.log(np.abs(discrimination))
    with np.errstate(over="ignore", under="ignore"):
        result += logits
        right = logits > -36.0
        if np.any(right):
            z = np.minimum(logits[right], _MAX_DOUBLE_EXP_INPUT)
            power = np.exp(z)
            result[right] = (
                2.0 * np.log(np.abs(discrimination[right]))
                + 2.0 * z
                - power
                - np.log(-np.expm1(-power))
            )
        np.exp(result, out=result)
    return result


def _double_exponential_information(
    logits: NDArray[np.float64], discrimination: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate both mirrored links from their separate success/failure tails."""
    power = _bounded_exponential(logits)
    information = -power
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        denominator = np.expm1(information)
        denominator *= -1.0
        np.exp(information, out=information)
        information *= power
        np.divide(power, denominator, out=power, where=denominator > 0.0)
        information *= power
        exceptional = (
            (information < np.finfo(float).tiny)
            & np.isfinite(logits)
            & (discrimination != 0.0)
        )
        information *= discrimination
        information *= discrimination
    if np.any(exceptional):
        information[exceptional] = _double_exponential_log_information(
            logits[exceptional],
            np.broadcast_to(discrimination, logits.shape)[exceptional],
        )
    return information


def _unidimensional_logits(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    *,
    item_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """Compute logits without losing finite offsets before slope rescaling."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if item_indices is None:
            logits = theta - difficulty
            logits *= discrimination
        else:
            logits = theta - difficulty[item_indices]
            logits *= discrimination[item_indices]
        recover = ~np.isfinite(logits)
        if np.any(recover):
            # For finite inputs, an overflowing difference can still yield a
            # finite logit with a small slope. Scale before subtracting here.
            if item_indices is None:
                a, t, b = (
                    np.broadcast_to(value, logits.shape)[recover]
                    for value in (discrimination, theta, difficulty)
                )
            else:
                indices = item_indices[recover]
                a, t, b = discrimination[indices], theta[recover], difficulty[indices]
            finite = np.isfinite(t) & np.isfinite(b) & (np.abs(a) <= 1.0)
            positions = np.flatnonzero(recover)[finite]
            logits.flat[positions] = a[finite] * t[finite] - a[finite] * b[finite]
    return logits


def _centered_logit_pairs(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Recenter suspicious dot products, resolving severe cancellation exactly."""
    products = _unidimensional_logits(theta, discrimination, difficulty[:, None])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        logits = np.sum(products, axis=1)
        bound = np.sum(np.abs(products), axis=1)
        error = 4.0 * np.finfo(float).eps * theta.shape[1] * bound
        precise = ~np.isfinite(logits) | (
            (bound > 1000.0) & (np.abs(logits) <= 2500.0 + error)
        )
    if np.any(precise):
        from fractions import Fraction

        slopes = np.broadcast_to(discrimination, theta.shape)
        for row in np.flatnonzero(precise):
            location = Fraction(float(difficulty[row]))
            value = sum(
                Fraction(float(slope)) * (Fraction(float(point)) - location)
                for slope, point in zip(slopes[row], theta[row], strict=True)
            )
            try:
                logits[row] = float(value)
            except OverflowError:
                logits[row] = -np.inf if value < 0 else np.inf
    return logits


def _multidimensional_logits(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    *,
    item_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """Keep BLAS for ordinary inputs and recenter affected respondent-item cells."""
    if item_indices is not None:
        discrimination = discrimination[item_indices]
        difficulty = difficulty[item_indices]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        offset = np.sum(discrimination, axis=-1) * difficulty
        if item_indices is not None:
            logits = np.einsum("ij,ij->i", theta, discrimination) - offset
        else:
            logits = np.dot(theta, discrimination.T) - offset
        theta_scale = np.max(np.abs(theta), initial=0.0)
        parameter_scale = np.max(np.abs(discrimination), initial=0.0) * theta.shape[1]
        max_bound = (
            theta_scale + np.max(np.abs(difficulty), initial=0.0)
        ) * parameter_scale
        if max_bound <= 1000.0:
            return logits
        row_scale = np.max(np.abs(theta), axis=1)
        slope_bound = np.sum(np.abs(discrimination), axis=-1)
        if logits.ndim == 2:
            row_scale = row_scale[:, None]
        bound = (row_scale + np.abs(difficulty)) * slope_bound
        error = 4.0 * np.finfo(float).eps * theta.shape[1] * bound
        recover = ~np.isfinite(logits) | (
            (bound > 1000.0) & (np.abs(logits) <= 2500.0 + error)
        )
    # Outside this range both the sigmoid and Fisher-information tails have
    # saturated even at the largest finite slope. Only uncertain cells need
    # centered respondent-by-factor buffers, whose storage is bounded here.
    positions = np.flatnonzero(recover)
    per_block = max(1, _LOGISTIC_CURVE_CHUNK_ELEMENTS // theta.shape[1])
    for start in range(0, positions.size, per_block):
        selected = positions[start : start + per_block]
        if logits.ndim == 2:
            rows, items = np.divmod(selected, logits.shape[1])
            slopes = discrimination[items]
            locations = difficulty[items]
        else:
            rows = selected
            if item_indices is None:
                slopes = np.broadcast_to(discrimination, (rows.size, theta.shape[1]))
                locations = np.full(rows.size, difficulty)
            else:
                slopes, locations = discrimination[rows], difficulty[rows]
        points = theta[rows]
        finite = (
            np.all(np.isfinite(points), axis=1)
            & np.all(np.isfinite(slopes), axis=1)
            & np.isfinite(locations)
        )
        if np.any(finite):
            logits.flat[selected[finite]] = _centered_logit_pairs(
                points[finite], slopes[finite], locations[finite]
            )
    return logits


def _double_exponential_curve(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    *,
    negative: bool,
    information: bool,
    item_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    logits = _unidimensional_logits(
        theta, discrimination, difficulty, item_indices=item_indices
    )
    if negative:
        logits *= -1.0
    if information:
        if item_indices is not None:
            discrimination = discrimination[item_indices]
        return _double_exponential_information(logits, discrimination)
    probability = _bounded_exponential(logits)
    probability *= -1.0
    with np.errstate(under="ignore"):
        if negative:
            np.exp(probability, out=probability)
        else:
            np.expm1(probability, out=probability)
            probability *= -1.0
    return probability


def _logistic_information(
    logits: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    guessing: NDArray[np.float64] | None = None,
    upper: NDArray[np.float64] | None = None,
    *,
    norm_factor: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Preserve both logistic tails and defer slope scaling until after reduction."""
    if guessing is None:
        return _scaled_information(logits, discrimination, norm_factor=norm_factor)

    from scipy.special import expit

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        width = 1.0 - guessing if upper is None else upper - guessing
        success = expit(logits)
        failure = expit(-logits)
        information = success * failure
        information *= width
        information *= discrimination
        np.square(information, out=information)
        success *= width
        success += guessing
        failure *= width
        if upper is not None:
            failure += 1.0 - upper
        success *= failure
        exceptional = (information < np.finfo(float).tiny) | (
            success < np.finfo(float).tiny
        )
        exceptional &= width > 0.0
        np.divide(information, success, out=information, where=success > 0.0)
        exceptional &= np.isfinite(logits) & (discrimination != 0.0)
    if np.any(exceptional):
        z = logits[exceptional]
        information[exceptional] = _five_pl_log_information(
            z,
            np.broadcast_to(discrimination, logits.shape)[exceptional],
            np.broadcast_to(guessing, logits.shape)[exceptional],
            np.ones_like(z)
            if upper is None
            else np.broadcast_to(upper, logits.shape)[exceptional],
            np.ones_like(z),
        )
    return information


def _unipolar_curve(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    *,
    information: bool,
    item_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    logits = _unidimensional_logits(
        theta, discrimination, difficulty, item_indices=item_indices
    )
    result = _sigmoid_derivative(logits)
    # Division can round a nearly maximal probability one ulp above 0.25.
    np.minimum(result, 0.25, out=result)
    if not information:
        return result
    if item_indices is not None:
        discrimination = discrimination[item_indices]
    exceptional = (result < np.finfo(float).tiny) & np.isfinite(logits)
    with np.errstate(over="ignore", under="ignore"):
        # I = a² P tanh²(z/2) / (1-P). The tanh form retains the small
        # derivative near z=0, where 1-2*sigmoid(z) would cancel to zero.
        scaled_slope = logits * 0.5
        np.tanh(scaled_slope, out=scaled_slope)
        scaled_slope *= discrimination
        result /= 1.0 - result
        result *= scaled_slope
        result *= scaled_slope
        if np.any(exceptional):
            # In either far tail, I = a² exp(-|z|) to float precision.
            a = np.broadcast_to(discrimination, logits.shape)[exceptional]
            result[exceptional] = np.exp(2.0 * np.log(a) - np.abs(logits[exceptional]))
    return result


def _log_powered_sigmoid(
    logits: NDArray[np.float64], asymmetry: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Keep sigmoid powers accurate when the unpowered sigmoid rounds to 0 or 1."""
    log_power = np.abs(logits)
    np.negative(log_power, out=log_power)
    with np.errstate(over="ignore", under="ignore"):
        # log(sigmoid(z)) = min(z, 0) - log1p(exp(-abs(z))). NumPy's
        # vectorized elementary functions also keep paired queries inexpensive.
        np.exp(log_power, out=log_power)
        np.log1p(log_power, out=log_power)
        np.subtract(np.minimum(logits, 0.0), log_power, out=log_power)
        log_power *= asymmetry
        tail = logits > 700.0
        if np.any(tail):
            # Combine the exponent with the shape parameter before exponentiating
            # so a large shape can recover an otherwise underflowing tail.
            shape = np.broadcast_to(asymmetry, logits.shape)[tail]
            log_power[tail] = -np.exp(np.log(shape) - logits[tail])
    return log_power


def _five_pl_probability(
    logits: NDArray[np.float64],
    guessing: NDArray[np.float64],
    upper: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
) -> NDArray[np.float64]:
    probability = _log_powered_sigmoid(logits, asymmetry)
    with np.errstate(under="ignore"):
        np.exp(probability, out=probability)
    probability *= upper - guessing
    probability += guessing
    return probability


def _five_pl_log_information(
    logits: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    guessing: NDArray[np.float64],
    upper: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    *,
    log_power: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Evaluate Fisher information with separate success and failure tails."""
    from scipy.special import log_expit

    if log_power is None:
        log_power = _log_powered_sigmoid(logits, asymmetry)
    width = upper - guessing
    usable = (
        (np.isfinite(logits) | np.isneginf(logits))
        & np.isfinite(log_power)
        & (discrimination != 0.0)
        & (width > 0.0)
    )
    information = np.zeros_like(logits)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_width = np.log(width)
        log_shape = np.log(asymmetry)
        log_failure_power = np.log(-np.expm1(log_power))
        rounded = log_power > -np.finfo(float).tiny
        if np.any(rounded):
            z = logits[rounded]
            log_softplus = np.where(z > 36.0, -z, np.log(-log_expit(z)))
            log_failure_power[rounded] = (
                np.broadcast_to(log_shape, logits.shape)[rounded] + log_softplus
            )
        log_success = np.logaddexp(np.log(guessing), log_width + log_power)
        log_failure = np.logaddexp(np.log1p(-upper), log_width + log_failure_power)
        log_slope = np.log(np.abs(discrimination)) + log_shape + log_expit(-logits)
        log_information = (
            2.0 * (log_width + log_slope)
            + log_power
            + (log_power - log_success)
            - log_failure
        )
        np.exp(log_information, out=information, where=usable)
    return information


def _five_pl_curve(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64],
    upper: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    *,
    information: bool = False,
    item_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """Evaluate a 5PL curve while recovering representable overflowed products."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if item_indices is None:
            logits = theta - difficulty
            logits *= discrimination
        else:
            logits = theta - difficulty[item_indices]
            logits *= discrimination[item_indices]
    exceptional = ~np.isfinite(logits)
    positions = np.empty(0, dtype=np.intp)
    if np.any(exceptional):
        if item_indices is None:
            arguments = [
                np.broadcast_to(value, logits.shape)[exceptional]
                for value in (
                    theta,
                    discrimination,
                    difficulty,
                    guessing,
                    upper,
                    asymmetry,
                )
            ]
        else:
            selected = item_indices[exceptional]
            arguments = [theta[exceptional]] + [
                value[selected]
                for value in (discrimination, difficulty, guessing, upper, asymmetry)
            ]
        finite = np.logical_and.reduce([np.isfinite(value) for value in arguments])
        positions = np.flatnonzero(exceptional)[finite]
        selected_theta, a, b, c, d, e = [value[finite] for value in arguments]
        # Halve the difference before subtracting, then combine binary exponents
        # so neither the difference nor a product needs to overflow prematurely.
        difference = 0.5 * selected_theta - 0.5 * b
        slope_fraction, slope_exponent = np.frexp(a)
        delta_fraction, delta_exponent = np.frexp(difference)
        with np.errstate(over="ignore", under="ignore"):
            recovered = np.ldexp(
                slope_fraction * delta_fraction, slope_exponent + delta_exponent + 1
            )
        logits.flat[positions] = recovered
        negative = np.isneginf(recovered)
        positions = positions[negative]
        a, c, d, e = (value[negative] for value in (a, c, d, e))
        if positions.size:
            shape_fraction, shape_exponent = np.frexp(e)
            with np.errstate(over="ignore", under="ignore"):
                # In this tail log(sigmoid(z)) equals z to floating-point
                # precision. Apply the shape before materializing the product.
                log_power = np.ldexp(
                    slope_fraction[negative]
                    * delta_fraction[negative]
                    * shape_fraction,
                    slope_exponent[negative]
                    + delta_exponent[negative]
                    + shape_exponent
                    + 1,
                )
    del exceptional
    if item_indices is not None:
        guessing = guessing[item_indices]
        upper = upper[item_indices]
        asymmetry = asymmetry[item_indices]
        if information:
            discrimination = discrimination[item_indices]
    result = (
        _five_pl_information(logits, discrimination, guessing, upper, asymmetry)
        if information
        else _five_pl_probability(logits, guessing, upper, asymmetry)
    )
    if positions.size:
        if information:
            result.flat[positions] = _five_pl_log_information(
                np.full(positions.size, -np.inf), a, c, d, e, log_power=log_power
            )
        else:
            with np.errstate(under="ignore"):
                np.exp(log_power, out=log_power)
            log_power *= d - c
            log_power += c
            result.flat[positions] = log_power
    return result


def _five_pl_information(
    logits: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    guessing: NDArray[np.float64],
    upper: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reuse probability buffers, resolving underflowed cells in log space."""
    from scipy.special import expit

    width = upper - guessing
    failure = _log_powered_sigmoid(logits, asymmetry)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        power = np.exp(failure)
        np.expm1(failure, out=failure)
        failure *= -width
        failure += 1.0 - upper
        success = power * width
        success += guessing
        information = expit(-logits)
        information *= asymmetry
        # The product of the powered curve and its log derivative is bounded
        # by one. Multiply these before the slope to avoid intermediate overflow.
        information *= power
        exceptional = (power < np.finfo(float).tiny) | (
            information < np.finfo(float).tiny
        )
        information *= discrimination
        information *= width
        np.square(information, out=information)
        success *= failure
        exceptional |= (information < np.finfo(float).tiny) | (
            success < np.finfo(float).tiny
        )
        exceptional &= np.isfinite(logits) & (discrimination != 0.0) & (width > 0.0)
        del power
        np.divide(information, success, out=information, where=success > 0.0)
    if np.any(exceptional):
        information[exceptional] = _five_pl_log_information(
            logits[exceptional],
            np.broadcast_to(discrimination, logits.shape)[exceptional],
            np.broadcast_to(guessing, logits.shape)[exceptional],
            np.broadcast_to(upper, logits.shape)[exceptional],
            np.broadcast_to(asymmetry, logits.shape)[exceptional],
        )
    return information


class _ParameterizedDichotomousModel(DichotomousItemModel):
    """Shared parameter-domain validation for dichotomous response curves."""

    _requires_positive_discrimination = False

    def _validate_parameter_state(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> None:
        for name, values in parameters.items():
            if not np.all(np.isfinite(values)):
                raise MirtValidationError(
                    f"{name} must contain only finite values",
                    parameter=name,
                    value=values,
                    expected="finite values",
                )

        discrimination = parameters["discrimination"]
        if self._requires_positive_discrimination and np.any(discrimination <= 0.0):
            raise MirtValidationError(
                "discrimination must be strictly positive",
                parameter="discrimination",
                value=discrimination,
                expected="> 0",
            )

        guessing = parameters.get("guessing")
        if guessing is not None and np.any((guessing < 0.0) | (guessing >= 1.0)):
            raise MirtValidationError(
                "guessing must be in [0, 1)",
                parameter="guessing",
                value=guessing,
                expected="[0, 1)",
            )

        upper = parameters.get("upper")
        if upper is not None:
            if np.any((upper < 0.0) | (upper > 1.0)):
                raise MirtValidationError(
                    "upper must be in [0, 1]",
                    parameter="upper",
                    value=upper,
                    expected="[0, 1]",
                )
            if guessing is not None and np.any(guessing > upper):
                raise MirtValidationError(
                    "guessing cannot exceed upper",
                    parameter="guessing",
                    value=guessing,
                    expected="guessing <= upper",
                )

        asymmetry = parameters.get("asymmetry")
        if asymmetry is not None and np.any(asymmetry <= 0.0):
            raise MirtValidationError(
                "asymmetry must be strictly positive",
                parameter="asymmetry",
                value=asymmetry,
                expected="> 0",
            )

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set parameters atomically after validating the complete model state."""
        candidate = {name: values.copy() for name, values in self._parameters.items()}
        for name, value in params.items():
            if name not in candidate:
                valid_params = ", ".join(candidate)
                raise MirtValidationError(
                    f"Unknown parameter: {name}. Valid parameters: {valid_params}",
                    parameter=name,
                    expected=valid_params,
                )

            value_array = np.asarray(value, dtype=np.float64)
            expected_shape = candidate[name].shape
            if value_array.shape != expected_shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}: expected {expected_shape}, "
                    f"got {value_array.shape}",
                    parameter=name,
                    value=value_array.shape,
                    expected=str(expected_shape),
                )
            candidate[name] = value_array.copy()

        self._validate_parameter_state(candidate)
        self._parameters = candidate
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one item parameter while preserving the model's domain."""
        item_idx = self._validate_item_idx(item_idx)
        if param_name not in self._parameters:
            valid_params = ", ".join(self._parameters)
            raise MirtValidationError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid_params}",
                parameter=param_name,
                expected=valid_params,
            )

        current = self._parameters[param_name]
        value_array = np.asarray(value, dtype=np.float64)
        expected_shape = current.shape[1:]
        if value_array.shape != expected_shape:
            expected = "scalar" if not expected_shape else str(expected_shape)
            raise MirtValidationError(
                f"{param_name} for one item must have shape {expected}",
                parameter=param_name,
                value=value_array.shape,
                expected=expected,
            )

        updated = current.copy()
        updated[item_idx] = value_array
        self.set_parameters(**{param_name: updated})

    def _validate_item_idx(self, item_idx: int) -> int:
        if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
            item_idx, (int, np.integer)
        ):
            raise IndexError("item_idx must be an integer")
        item_idx = int(item_idx)
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")
        return item_idx

    def _evaluate_logistic(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
        *,
        information: bool = False,
        item_indices: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Share bounded full, single-item, and paired evaluation across 1PL–4PL."""
        if item_indices is None:
            theta = self._ensure_theta_2d(theta)
        slope = self._parameters["discrimination"]
        location = self._parameters["difficulty"]
        guessing = self._parameters.get("guessing")
        upper = self._parameters.get("upper")
        all_items = item_idx is None and item_indices is None
        width = self.n_items if all_items else 1
        shape = (theta.shape[0], self.n_items) if all_items else (theta.shape[0],)
        if item_idx is not None:
            slope, location = slope[item_idx], location[item_idx]
            if guessing is not None:
                guessing = guessing[item_idx]
            if upper is not None:
                upper = upper[item_idx]
        norm_factor = None
        magnitude = slope
        if information and self.n_factors != 1:
            # Keep the squared norm factored until after the logistic tail
            # reduction, including for exactly zero slope vectors.
            magnitude = np.max(np.abs(slope), axis=-1)
            normalized = np.divide(
                slope,
                magnitude[..., None],
                out=np.zeros_like(slope),
                where=magnitude[..., None] > 0.0,
            )
            norm_factor = np.sum(normalized**2, axis=-1)

        def evaluate(
            points: NDArray[np.float64],
            selected: NDArray[np.intp] | None,
        ) -> NDArray[np.float64]:
            if self.n_factors == 1:
                abilities = points[:, 0, None] if all_items else points[:, 0]
                logits = _unidimensional_logits(
                    abilities, slope, location, item_indices=selected
                )
            else:
                logits = _multidimensional_logits(
                    points, slope, location, item_indices=selected
                )
            c, d, a, factor = guessing, upper, magnitude, norm_factor
            if selected is not None:
                if c is not None:
                    c = c[selected]
                if d is not None:
                    d = d[selected]
                if information:
                    a = a[selected]
                    if factor is not None:
                        factor = factor[selected]
            if information:
                return _logistic_information(logits, a, c, d, norm_factor=factor)
            return _logistic_probability(logits, c, d)

        rows_per_block = max(
            1, _LOGISTIC_CURVE_CHUNK_ELEMENTS // max(width, self.n_factors)
        )
        if theta.shape[0] <= rows_per_block:
            return evaluate(theta, item_indices)
        result = np.empty(shape)
        for start in range(0, theta.shape[0], rows_per_block):
            rows = slice(start, start + rows_per_block)
            result[rows] = evaluate(
                theta[rows], None if item_indices is None else item_indices[rows]
            )
        return result


class TwoParameterLogistic(_ParameterizedDichotomousModel):
    model_name = "2PL"
    n_params_per_item = 2
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_logistic(theta, item_idx)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in bounded vectorized batches."""
        theta, indices = self._prepare_probability_pairs(theta, item_indices)
        return self._evaluate_logistic(theta, None, item_indices=indices)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_logistic(theta, item_idx, information=True)


class OneParameterLogistic(TwoParameterLogistic):
    model_name = "1PL"
    n_params_per_item = 1
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("1PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["discrimination"] = np.zeros_like(self.discrimination, dtype=np.bool_)
        return masks

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "discrimination":
            canonical.fill(1.0)
        return canonical

    def set_parameters(self, **params: NDArray[np.float64]) -> "OneParameterLogistic":
        if "discrimination" in params:
            raise ValueError("Cannot set discrimination in 1PL model (fixed to 1)")
        return super().set_parameters(**params)

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        if param_name == "discrimination":
            self._validate_item_idx(item_idx)
            value_array = np.asarray(value, dtype=np.float64)
            if value_array.ndim == 0 and float(value_array) == 1.0:
                return
            raise ValueError("Cannot set discrimination in 1PL model (fixed to 1)")
        super().set_item_parameter(item_idx, param_name, value)


class ThreeParameterLogistic(_ParameterizedDichotomousModel):
    model_name = "3PL"
    n_params_per_item = 3
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("3PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)
        self._parameters["guessing"] = np.full(self.n_items, 0.2)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        return self._parameters["guessing"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_logistic(theta, item_idx)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in bounded vectorized batches."""
        theta, indices = self._prepare_probability_pairs(theta, item_indices)
        return self._evaluate_logistic(theta, None, item_indices=indices)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_logistic(theta, item_idx, information=True)


class FourParameterLogistic(_ParameterizedDichotomousModel):
    model_name = "4PL"
    n_params_per_item = 4
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("4PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)
        self._parameters["guessing"] = np.full(self.n_items, 0.2)
        self._parameters["upper"] = np.ones(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        return self._parameters["guessing"]

    @property
    def upper(self) -> NDArray[np.float64]:
        return self._parameters["upper"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_logistic(theta, item_idx)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in bounded vectorized batches."""
        theta, indices = self._prepare_probability_pairs(theta, item_indices)
        return self._evaluate_logistic(theta, None, item_indices=indices)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_logistic(theta, item_idx, information=True)


Rasch = OneParameterLogistic

ThreeParameterLogisticUpper = FourParameterLogistic


class _UnidimensionalCurveModel(_ParameterizedDichotomousModel):
    """Shared bounded evaluation for two-parameter unidimensional curves."""

    n_params_per_item = 2
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError(
                f"{self.model_name} model only supports unidimensional analysis"
            )
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @abstractmethod
    def _evaluate_block(
        self,
        theta: NDArray[np.float64],
        discrimination: NDArray[np.float64],
        difficulty: NDArray[np.float64],
        *,
        information: bool,
        item_indices: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]: ...

    def _evaluate_curve(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
        *,
        information: bool = False,
        item_indices: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        slope = self._parameters["discrimination"]
        location = self._parameters["difficulty"]
        if item_indices is not None:
            points = theta[:, 0]
            shape = (theta.shape[0],)
            width = 1
        elif item_idx is None:
            points = theta[:, 0, None]
            shape = (theta.shape[0], self.n_items)
            width = self.n_items
        else:
            points = theta[:, 0]
            slope, location = slope[item_idx], location[item_idx]
            shape = (theta.shape[0],)
            width = 1
        rows_per_block = max(1, _UNIDIMENSIONAL_CURVE_CHUNK_ELEMENTS // width)
        if theta.shape[0] <= rows_per_block:
            return self._evaluate_block(
                points,
                slope,
                location,
                information=information,
                item_indices=item_indices,
            )
        result = np.empty(shape)
        for start in range(0, theta.shape[0], rows_per_block):
            rows = slice(start, start + rows_per_block)
            result[rows] = self._evaluate_block(
                points[rows],
                slope,
                location,
                information=information,
                item_indices=None if item_indices is None else item_indices[rows],
            )
        return result

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_curve(theta, item_idx)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in bounded vectorized batches."""
        theta, indices = self._prepare_probability_pairs(theta, item_indices)
        return self._evaluate_curve(theta, None, item_indices=indices)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_curve(theta, item_idx, information=True)


class UnipolarLogLogistic(_UnidimensionalCurveModel):
    """Unipolar Log-Logistic (ULL) model for dichotomous items.

    This implementation uses a symmetric, bell-shaped response curve.
    Endorsement is greatest at the item location and declines in either
    direction. The maximum response probability is 0.25.

    Parameters
    ----------
    n_items : int
        Number of items
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination parameters (must be positive)
    difficulty : ndarray
        Item difficulty/location parameters

    Notes
    -----
    The ULL probability function is:

        P(X=1|θ) = exp(a(θ - b)) / (1 + exp(a(θ - b)))^2

    This is the derivative of the logistic sigmoid with respect to its
    logit. Its peak is at θ = b, and it approaches zero in both tails.
    """

    model_name = "ULL"
    _requires_positive_discrimination = True

    def _evaluate_block(
        self,
        theta: NDArray[np.float64],
        discrimination: NDArray[np.float64],
        difficulty: NDArray[np.float64],
        *,
        information: bool,
        item_indices: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        return _unipolar_curve(
            theta,
            discrimination,
            difficulty,
            information=information,
            item_indices=item_indices,
        )


class FiveParameterLogistic(_ParameterizedDichotomousModel):
    """Five-Parameter Logistic (5PL) model with asymmetric curves.

    The 5PL model extends the 4PL with an asymmetry parameter that allows
    the IRF to have different slopes in the lower and upper regions.
    This is useful when item characteristics vary across the ability range.

    Parameters
    ----------
    n_items : int
        Number of items
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination (slope) parameters
    difficulty : ndarray
        Item difficulty (location) parameters
    guessing : ndarray
        Lower asymptote (guessing) parameters
    upper : ndarray
        Upper asymptote parameters
    asymmetry : ndarray
        Asymmetry parameters (> 1 steeper on right, < 1 steeper on left)

    Notes
    -----
    The 5PL probability function is:

        P(X=1|θ) = c + (d - c) / (1 + exp(-a(θ - b)))^e

    where e is the asymmetry parameter.

    References
    ----------
    Reise, S. P., & Waller, N. G. (2003). How many IRT parameters does it
        take to model psychopathology items? Psychological Methods.
    """

    model_name = "5PL"
    n_params_per_item = 5
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("5PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)
        self._parameters["guessing"] = np.full(self.n_items, 0.2)
        self._parameters["upper"] = np.ones(self.n_items)
        self._parameters["asymmetry"] = np.ones(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        return self._parameters["guessing"]

    @property
    def upper(self) -> NDArray[np.float64]:
        return self._parameters["upper"]

    @property
    def asymmetry(self) -> NDArray[np.float64]:
        return self._parameters["asymmetry"]

    def _evaluate_curve(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
        *,
        information: bool = False,
        item_indices: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        parameters = [
            self._parameters[name]
            for name in (
                "discrimination",
                "difficulty",
                "guessing",
                "upper",
                "asymmetry",
            )
        ]
        if item_indices is not None:
            points = theta[:, 0]
            shape = (theta.shape[0],)
            width = 1
        elif item_idx is None:
            points = theta[:, 0, None]
            shape = (theta.shape[0], self.n_items)
            width = self.n_items
        else:
            points = theta[:, 0]
            parameters = [parameter[item_idx] for parameter in parameters]
            shape = (theta.shape[0],)
            width = 1
        rows_per_block = max(1, _FIVE_PL_CURVE_CHUNK_ELEMENTS // width)
        if theta.shape[0] <= rows_per_block:
            return _five_pl_curve(
                points, *parameters, information=information, item_indices=item_indices
            )
        result = np.empty(shape)
        for start in range(0, theta.shape[0], rows_per_block):
            rows = slice(start, start + rows_per_block)
            result[rows] = _five_pl_curve(
                points[rows],
                *parameters,
                information=information,
                item_indices=None if item_indices is None else item_indices[rows],
            )
        return result

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_curve(theta, item_idx)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in bounded vectorized batches."""
        theta, indices = self._prepare_probability_pairs(theta, item_indices)
        return self._evaluate_curve(theta, None, item_indices=indices)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        return self._evaluate_curve(theta, item_idx, information=True)


class _DoubleExponentialModel(_UnidimensionalCurveModel):
    """Select the direction of the mirrored CLL and NLL response curves."""

    _negative_loglog = False

    def _evaluate_block(
        self,
        theta: NDArray[np.float64],
        discrimination: NDArray[np.float64],
        difficulty: NDArray[np.float64],
        *,
        information: bool,
        item_indices: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        return _double_exponential_curve(
            theta,
            discrimination,
            difficulty,
            negative=self._negative_loglog,
            information=information,
            item_indices=item_indices,
        )


class ComplementaryLogLog(_DoubleExponentialModel):
    """Complementary Log-Log (CLL) model for dichotomous items.

    The CLL model uses an asymmetric link function instead of the
    symmetric logistic. This is useful when the probability curve
    should approach 0 and 1 at different rates.

    Parameters
    ----------
    n_items : int
        Number of items
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination parameters
    difficulty : ndarray
        Item difficulty parameters

    Notes
    -----
    The CLL probability function is:

        P(X=1|θ) = 1 - exp(-exp(a(θ - b)))

    The CLL function approaches 0 slowly and 1 quickly.

    For slow approach to 1 and fast to 0, use the negative-log-log
    (NLL) variant: P = exp(-exp(-a(θ - b)))
    """

    model_name = "CLL"


class NegativeLogLog(_DoubleExponentialModel):
    """Negative Log-Log (NLL) model for dichotomous items.

    The NLL model is the mirror image of CLL, approaching 1 slowly
    and 0 quickly.

    Notes
    -----
    The NLL probability function is:

        P(X=1|θ) = exp(-exp(-a(θ - b)))
    """

    model_name = "NLL"
    _negative_loglog = True
