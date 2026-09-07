"""The prior is independent of the model-input tree used for filtering."""

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random, tree

from cuthbert import filter
from cuthbert.discrete import filter as discrete
from cuthbert.ensemble_kalman import ensemble_kalman_filter as enkf
from cuthbert.gaussian import kalman, moments, taylor
from cuthbert.smc import marginal_particle_filter, particle_filter
from cuthbertlib.resampling import no_resampling
from cuthbertlib.stats.multivariate_normal import logpdf


def gaussian_filter(method, mean, associative):
    chol_cov = jnp.eye(1)
    if method == "kalman":
        return kalman.build_filter(
            mean,
            chol_cov,
            lambda inputs: (jnp.eye(1), jnp.zeros(1), jnp.eye(1)),
            lambda inputs: (jnp.eye(1), jnp.zeros(1), jnp.eye(1), inputs["y"]),
        )
    if method == "moments":
        return moments.build_filter(
            mean,
            chol_cov,
            lambda state, inputs: (lambda x: (x, jnp.eye(1)), jnp.zeros(1)),
            lambda state, inputs: (
                lambda x: (x, jnp.eye(1)),
                jnp.zeros(1),
                inputs["y"],
            ),
            associative=associative,
        )
    return taylor.build_filter(
        init_log_density=lambda x: logpdf(x, mean, chol_cov),
        init_linearization_point=jnp.zeros(1),
        get_dynamics_log_density=lambda state, inputs: (
            lambda prev, curr: logpdf(curr, prev, jnp.eye(1)),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
        get_observation_func=lambda state, inputs: (
            lambda x, y: logpdf(y, x, jnp.eye(1)),
            jnp.zeros(1),
            inputs["y"],
        ),
        associative=associative,
    )


@pytest.mark.parametrize("method", ["kalman", "moments", "taylor"])
@pytest.mark.parametrize("associative", [False, True])
def test_explicit_gaussian_prior_jit_and_grad(method, associative):
    # Build inside jit so the explicit prior parameters are tracers. Step inputs
    # contain only observations, and cannot be used to reconstruct the prior.
    def posterior(initial_mean):
        obj = gaussian_filter(method, initial_mean, associative)
        prior = obj.init_prepare()
        assert prior.model_inputs is None
        result = filter(obj, {"y": jnp.array([[4.0]])}, prior, parallel=associative)
        chex.assert_shape(result.mean, (2, 1))
        return result.mean[-1, 0]

    mean = jnp.array([1.0])
    chex.assert_trees_all_close(jax.jit(posterior)(mean), 3.0, atol=1e-5)
    chex.assert_trees_all_close(
        jax.jit(jax.grad(posterior))(mean), jnp.array([1 / 3]), atol=1e-5
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_explicit_discrete_prior(parallel):
    prior_dist = jnp.array([0.25, 0.75])
    obj = discrete.build_filter(
        init_dist=prior_dist,
        get_trans_matrix=lambda inputs: jnp.eye(2),
        get_obs_lls=lambda inputs: inputs["log_likelihoods"],
    )
    prior = jax.jit(obj.init_prepare)()
    assert prior.model_inputs is None
    inputs = {"log_likelihoods": jnp.log(jnp.array([[0.8, 0.2]]))}
    states = jax.jit(lambda state: filter(obj, inputs, state, parallel=parallel))(prior)
    chex.assert_trees_all_close(states.dist[0], prior_dist)
    chex.assert_trees_all_close(states.dist[1], jnp.array([4 / 7, 3 / 7]))


@pytest.mark.parametrize("method", [particle_filter, marginal_particle_filter, enkf])
def test_key_only_initial_sampler(method):
    mean = jnp.array([2.0, -1.0], dtype=jnp.float32)

    def init_sample(key):
        return mean + random.normal(key, mean.shape, dtype=mean.dtype)

    if method is enkf:
        obj = method.build_filter(
            init_sample,
            lambda inputs: lambda x, key: x,
            lambda inputs: (lambda x: x, jnp.eye(2), inputs["y"]),
            n_particles=8,
            store_predicted_ensemble=True,
        )

        def get_particles(state):
            return state.ensemble
    else:
        obj = method.build_filter(
            init_sample,
            lambda key, state, inputs: state,
            lambda prev, curr, inputs: jnp.array(0.0),
            n_filter_particles=8,
            resampling_fn=no_resampling.resampling,
        )

        def get_particles(state):
            return state.particles

    with pytest.raises(ValueError, match="PRNG key"):
        obj.init_prepare()
    prior = jax.jit(obj.init_prepare)(key=random.key(0))
    assert prior.model_inputs is None
    expected = jax.vmap(init_sample)(random.split(random.key(0), 8))
    chex.assert_trees_all_close(get_particles(prior), expected, atol=1e-6)
    inputs = {"y": jnp.full((2, 2), jnp.nan)}
    # Shape inference must preserve the sampler dtype even when it differs from
    # JAX's default floating-point dtype.
    previous_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        prepared = obj.filter_prepare(
            tree.map(lambda x: x[0], inputs), key=random.key(1)
        )
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    assert get_particles(prepared).dtype == mean.dtype
    states = jax.jit(lambda state: filter(obj, inputs, state, key=random.key(2)))(prior)
    chex.assert_shape(get_particles(states), (3, 8, 2))
    chex.assert_trees_all_close(get_particles(states)[-1], expected, atol=1e-6)
