# mirt

**Multidimensional Item Response Theory for Python**

A comprehensive Python implementation of Item Response Theory (IRT) models, inspired by R's [mirt](https://github.com/philchalmers/mirt) package.

## Features

- **Dichotomous Models**: 1PL (Rasch), 2PL, 3PL, 4PL
- **Polytomous Models**: GRM, GPCM, PCM, NRM
- **Estimation**: EM algorithm with Gauss-Hermite quadrature
- **Scoring**: EAP, MAP, ML methods for person ability estimation
- **Diagnostics**: Item fit, person fit statistics
- **Simulation**: Generate response data from IRT models
- **Multiple Groups**: Basic multiple group analysis

## Installation

```bash
pip install -e .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import mirt
import numpy as np

# Simulate response data
responses = mirt.simdata(model='2PL', n_persons=500, n_items=20, seed=42)

# Fit a 2PL model
result = mirt.fit_mirt(responses, model='2PL')

# View results
print(result.summary())

# Get item parameters
params = result.coef()
print(params)

# Score respondents
scores = mirt.fscores(result, responses, method='EAP')
print(scores.to_dataframe().head())
```

## Supported Models

### Dichotomous (Binary) Models

| Model | Description | Parameters |
|-------|-------------|------------|
| 1PL/Rasch | One-parameter logistic | difficulty (b) |
| 2PL | Two-parameter logistic | discrimination (a), difficulty (b) |
| 3PL | Three-parameter logistic | a, b, guessing (c) |
| 4PL | Four-parameter logistic | a, b, c, upper asymptote (d) |

### Polytomous (Ordinal) Models

| Model | Description | Use Case |
|-------|-------------|----------|
| GRM | Graded Response Model | Likert scales |
| GPCM | Generalized Partial Credit | Partial credit items |
| PCM | Partial Credit Model | Rasch for polytomous |
| NRM | Nominal Response Model | Unordered categories |

## Examples

### Fitting Different Models

```python
# 1PL (Rasch) model
result_1pl = mirt.fit_mirt(responses, model='1PL')

# 3PL model with guessing
result_3pl = mirt.fit_mirt(responses, model='3PL')

# Graded Response Model for Likert data
likert_data = mirt.simdata(model='GRM', n_categories=5)
result_grm = mirt.fit_mirt(likert_data, model='GRM', n_categories=5)
```

### Person Scoring

```python
# Different scoring methods
eap_scores = mirt.fscores(result, responses, method='EAP')
map_scores = mirt.fscores(result, responses, method='MAP')
ml_scores = mirt.fscores(result, responses, method='ML')

# Access theta values
print(eap_scores.theta)
print(eap_scores.standard_error)
```

### Item Analysis

```python
# Item fit statistics
item_fit = mirt.itemfit(result, responses)
print(item_fit)

# Person fit statistics
person_fit = mirt.personfit(result, responses)
print(person_fit)
```

### Data Simulation

```python
# Simulate with specific parameters
a = np.random.lognormal(0, 0.3, size=20)
b = np.random.normal(0, 1, size=20)

responses = mirt.simdata(
    model='2PL',
    discrimination=a,
    difficulty=b,
    n_persons=1000,
    seed=42
)
```

## API Reference

### Main Functions

- `fit_mirt(data, model, ...)` - Fit an IRT model
- `fscores(model, responses, method)` - Compute person abilities
- `simdata(model, n_persons, n_items, ...)` - Simulate response data
- `itemfit(result, responses)` - Item fit statistics
- `personfit(result, responses)` - Person fit statistics

### Model Classes

- `TwoParameterLogistic` - 2PL model class
- `ThreeParameterLogistic` - 3PL model class
- `GradedResponseModel` - GRM model class
- etc.

## Comparison with R mirt

| Feature | R mirt | Python mirt |
|---------|--------|-------------|
| Dichotomous models | 1PL-4PL | 1PL-4PL |
| Polytomous models | GRM, GPCM, PCM, NRM | GRM, GPCM, PCM, NRM |
| Estimation | EM, MHRM | EM |
| Multidimensional | Full support | Basic 2PL |
| Bifactor | Yes | Planned |
| DIF | Yes | Basic |
| GUI | Shiny app | - |

## Dependencies

- numpy >= 1.21
- scipy >= 1.7
- pandas >= 1.3

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/mirt

# Formatting
black src tests
ruff check src tests
```

## License

MIT License - see [LICENSE](LICENSE)

## References

- Chalmers, R. P. (2012). mirt: A Multidimensional Item Response Theory Package for the R Environment. *Journal of Statistical Software*, 48(6), 1-29.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443-459.
