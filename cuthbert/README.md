# cuthbert

This folder contains the code for the main `cuthbert` package.

<!-- --8<-- [start:unified_interface] -->
All inference methods are implemented with the following unified interface. Pass the
prior parameters to `build_filter`, then call `init_prepare()` (or
`init_prepare(key=key)` for stochastic methods). Initialization does not take
`model_inputs`; these describe the subsequent filtering steps.

The initialization arguments to `build_filter` depend on the method:

| Method | Initialization arguments |
| --- | --- |
| Kalman and moments | `m0`, `chol_P0` |
| Taylor | `init_log_density`, `init_linearization_point` |
| Discrete | `init_dist` |
| Particle, marginal particle, and ensemble Kalman | `init_sample(key)`, with prior parameters captured in a closure or `functools.partial` |

Initial states have `model_inputs=None`. Offline filtering pads that field with
unused dummy values so the initial state can be stacked with subsequent states.

```python
from jax import tree
import cuthbert

# Define model_inputs
filter_model_inputs = ...

# Load inference method
kalman_filter = cuthbert.gaussian.kalman.build_filter(
    m0=m0,
    chol_P0=chol_P0,
    get_dynamics_params=get_dynamics_params,
    get_observation_params=get_observation_params,
)   # build_filter function takes all inference-specific arguments, swap this out for different inference methods.

# Online inference
state = kalman_filter.init_prepare()

for t in range(T):
    model_inputs_t = tree.map(lambda x: x[t], filter_model_inputs)
    prepare_state = kalman_filter.filter_prepare(model_inputs_t)
    state = kalman_filter.filter_combine(state, prepare_state)
```

Or for offline inference:

```python
kalman_smoother = cuthbert.gaussian.kalman.build_smoother(get_dynamics_params)

init_state = kalman_filter.init_prepare()
filter_states = cuthbert.filter(kalman_filter, filter_model_inputs, init_state)
smoother_states = cuthbert.smoother(kalman_smoother, filter_states, filter_model_inputs)
```
<!-- --8<-- [end:unified_interface] -->
