import numpy as np
from numpy.typing import NDArray

from mirt.models.base import PolytomousItemModel


class GradedResponseModel(PolytomousItemModel):
    model_name = "GRM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        max_cats = max(self._n_categories)
        thresholds = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                thresholds[i, : n_cat - 1] = np.linspace(-2, 2, n_cat - 1)

        self._parameters["thresholds"] = thresholds

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        return self._parameters["thresholds"]

    def cumulative_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        threshold_idx: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)

        a = self._parameters["discrimination"]
        b = self._parameters["thresholds"][item_idx, threshold_idx]

        if self.n_factors == 1:
            a_item = a[item_idx]
            z = a_item * (theta.ravel() - b)
        else:
            a_item = a[item_idx]
            z = np.dot(theta, a_item) - np.sum(a_item) * b

        return 1.0 / (1.0 + np.exp(-z))

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        if category == 0:
            return 1.0 - self.cumulative_probability(theta, item_idx, 0)
        elif category == n_cat - 1:
            return self.cumulative_probability(theta, item_idx, category - 1)
        else:
            p_upper = self.cumulative_probability(theta, item_idx, category - 1)
            p_lower = self.cumulative_probability(theta, item_idx, category)
            return p_upper - p_lower

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            probs = np.zeros((n_persons, n_cat))
            for k in range(n_cat):
                probs[:, k] = self.category_probability(theta, item_idx, k)
            return probs

        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            for k in range(n_cat):
                probs[:, i, k] = self.category_probability(theta, i, k)

        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        ll = np.zeros(n_persons)

        for i in range(self.n_items):
            for person in range(n_persons):
                resp = responses[person, i]
                if resp >= 0:
                    prob = self.category_probability(
                        theta[person : person + 1], i, resp
                    )
                    ll[person] += np.log(prob[0] + 1e-10)

        return ll

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._item_information(theta, item_idx)

        info = np.zeros(n_persons)
        for i in range(self.n_items):
            info += self._item_information(theta, i)

        return info

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            a_val = a[item_idx]
        else:
            a_val = np.sqrt(np.sum(a[item_idx] ** 2))

        probs = self.probability(theta, item_idx)

        cum_probs = np.zeros((n_persons, n_cat + 1))
        cum_probs[:, 0] = 1.0
        for k in range(n_cat - 1):
            cum_probs[:, k + 1] = self.cumulative_probability(theta, item_idx, k)
        cum_probs[:, n_cat] = 0.0

        info = np.zeros(n_persons)
        for k in range(n_cat):
            p_k = probs[:, k]
            p_star_k = cum_probs[:, k]
            p_star_k1 = cum_probs[:, k + 1]

            dp_k = a_val * (p_star_k * (1 - p_star_k) - p_star_k1 * (1 - p_star_k1))

            info += np.where(p_k > 1e-10, (dp_k**2) / p_k, 0.0)

        return info


class GeneralizedPartialCredit(PolytomousItemModel):
    model_name = "GPCM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        max_cats = max(self._n_categories)
        steps = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                steps[i, : n_cat - 1] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["steps"] = steps

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def steps(self) -> NDArray[np.float64]:
        return self._parameters["steps"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["discrimination"]
        b = self._parameters["steps"][item_idx]

        if self.n_factors == 1:
            a_item = a[item_idx]
            theta_1d = theta.ravel()
        else:
            a_item = a[item_idx]
            theta_1d = np.dot(theta, a_item)
            a_item = np.sqrt(np.sum(a_item**2))

        numerators = np.zeros((n_persons, n_cat))

        for k in range(n_cat):
            cumsum = 0.0
            for v in range(k):
                cumsum += a_item * (theta_1d - b[v])
            numerators[:, k] = np.exp(cumsum)

        denominator = numerators.sum(axis=1)

        return numerators[:, category] / denominator

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            probs = np.zeros((n_persons, n_cat))
            for k in range(n_cat):
                probs[:, k] = self.category_probability(theta, item_idx, k)
            return probs

        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            for k in range(n_cat):
                probs[:, i, k] = self.category_probability(theta, i, k)

        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        ll = np.zeros(n_persons)

        for i in range(self.n_items):
            for person in range(n_persons):
                resp = responses[person, i]
                if resp >= 0:
                    prob = self.category_probability(
                        theta[person : person + 1], i, resp
                    )
                    ll[person] += np.log(prob[0] + 1e-10)

        return ll

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._item_information(theta, item_idx)

        info = np.zeros(n_persons)
        for i in range(self.n_items):
            info += self._item_information(theta, i)

        return info

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_categories[item_idx]

        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            a_val = a[item_idx]
        else:
            a_val = np.sqrt(np.sum(a[item_idx] ** 2))

        probs = self.probability(theta, item_idx)

        categories = np.arange(n_cat)
        expected = np.sum(probs * categories, axis=1)

        expected_sq = np.sum(probs * (categories**2), axis=1)

        variance = expected_sq - expected**2

        return (a_val**2) * variance


class PartialCreditModel(GeneralizedPartialCredit):
    model_name = "PCM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("PCM only supports unidimensional analysis")
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)

        max_cats = max(self._n_categories)
        steps = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                steps[i, : n_cat - 1] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["steps"] = steps

    def set_parameters(self, **params: NDArray[np.float64]) -> "PartialCreditModel":
        if "discrimination" in params:
            raise ValueError("Cannot set discrimination in PCM (fixed to 1)")
        return super().set_parameters(**params)


class NominalResponseModel(PolytomousItemModel):
    model_name = "NRM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        max_cats = max(self._n_categories)

        if self.n_factors == 1:
            slopes = np.zeros((self.n_items, max_cats))
            for i, n_cat in enumerate(self._n_categories):
                slopes[i, 1:n_cat] = np.linspace(0.5, 1.5, n_cat - 1)
        else:
            slopes = np.zeros((self.n_items, max_cats, self.n_factors))
            for i, n_cat in enumerate(self._n_categories):
                for f in range(self.n_factors):
                    slopes[i, 1:n_cat, f] = np.linspace(0.5, 1.5, n_cat - 1)

        self._parameters["slopes"] = slopes

        intercepts = np.zeros((self.n_items, max_cats))
        for i, n_cat in enumerate(self._n_categories):
            intercepts[i, 1:n_cat] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["intercepts"] = intercepts

    @property
    def slopes(self) -> NDArray[np.float64]:
        return self._parameters["slopes"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        return self._parameters["intercepts"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["slopes"]
        c = self._parameters["intercepts"]

        numerators = np.zeros((n_persons, n_cat))

        for k in range(n_cat):
            if self.n_factors == 1:
                z = a[item_idx, k] * theta.ravel() + c[item_idx, k]
            else:
                z = np.dot(theta, a[item_idx, k]) + c[item_idx, k]
            numerators[:, k] = np.exp(z)

        denominator = numerators.sum(axis=1)

        return numerators[:, category] / denominator

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            probs = np.zeros((n_persons, n_cat))
            for k in range(n_cat):
                probs[:, k] = self.category_probability(theta, item_idx, k)
            return probs

        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            for k in range(n_cat):
                probs[:, i, k] = self.category_probability(theta, i, k)

        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        ll = np.zeros(n_persons)

        for i in range(self.n_items):
            for person in range(n_persons):
                resp = responses[person, i]
                if resp >= 0:
                    prob = self.category_probability(
                        theta[person : person + 1], i, resp
                    )
                    ll[person] += np.log(prob[0] + 1e-10)

        return ll

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._item_information(theta, item_idx)

        info = np.zeros(n_persons)
        for i in range(self.n_items):
            info += self._item_information(theta, i)

        return info

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["slopes"]
        probs = self.probability(theta, item_idx)

        if self.n_factors == 1:
            a_item = a[item_idx, :n_cat]

            expected_a = np.sum(probs * a_item, axis=1)

            expected_a_sq = np.sum(probs * (a_item**2), axis=1)

            info = expected_a_sq - expected_a**2
        else:
            info = np.zeros(n_persons)
            for f in range(self.n_factors):
                a_f = a[item_idx, :n_cat, f]
                expected_a = np.sum(probs * a_f, axis=1)
                expected_a_sq = np.sum(probs * (a_f**2), axis=1)
                info += expected_a_sq - expected_a**2

        return info
