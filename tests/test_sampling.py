"""CPU unit tests for sampling. No GPU, no model, no kernel."""

from __future__ import annotations

import math

import pytest
import torch

from engine.sampling import (
    SamplingParams,
    StopState,
    apply_repetition_penalty,
    make_generator,
    sample,
    top_p_filter,
)


def logits_from(probabilities: list[float]) -> torch.Tensor:
    """Logits whose softmax is exactly `probabilities`, so the maths is checkable."""
    return torch.log(torch.tensor(probabilities, dtype=torch.float32))


# -- params validation -----------------------------------------------------


def test_defaults_are_deterministic_and_penalty_free():
    params = SamplingParams()
    assert params.is_greedy
    assert params.top_p == 1.0
    assert params.repetition_penalty == 1.0
    assert params.stop_token_ids == set()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"max_tokens": 0},
        {"repetition_penalty": 0.0},
    ],
)
def test_invalid_params_are_refused(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_temperature_zero_is_greedy_but_tiny_temperature_is_not():
    assert SamplingParams(temperature=0.0).is_greedy
    assert not SamplingParams(temperature=0.5).is_greedy


# -- greedy ----------------------------------------------------------------


def test_greedy_picks_the_argmax():
    logits = torch.tensor([1.0, 5.0, 3.0])
    assert sample(logits, SamplingParams()).tolist() == [1]


def test_greedy_handles_a_batch():
    logits = torch.tensor([[1.0, 5.0, 3.0], [9.0, 0.0, 2.0]])
    assert sample(logits, SamplingParams()).tolist() == [1, 0]


def test_greedy_ignores_temperature_scaling_entirely():
    """Temperature cannot change an argmax, so greedy must not depend on it."""
    logits = torch.tensor([0.1, 0.2, 0.15])
    assert sample(logits, SamplingParams(temperature=0.0)).tolist() == [1]


# -- top-p -----------------------------------------------------------------


def test_top_p_keeps_the_crossing_token():
    probabilities = torch.tensor([0.5, 0.3, 0.15, 0.05])
    filtered = top_p_filter(probabilities, 0.6)
    # 0.5 alone is under 0.6, so the 0.3 token is the one that crosses and stays.
    assert filtered[0] > 0 and filtered[1] > 0
    assert filtered[2] == 0 and filtered[3] == 0


def test_top_p_renormalises_to_a_valid_distribution():
    probabilities = torch.tensor([0.5, 0.3, 0.15, 0.05])
    filtered = top_p_filter(probabilities, 0.6)
    assert math.isclose(float(filtered.sum()), 1.0, rel_tol=1e-6)
    assert math.isclose(float(filtered[0] / filtered[1]), 0.5 / 0.3, rel_tol=1e-6)


def test_top_p_of_one_is_a_no_op():
    probabilities = torch.tensor([0.5, 0.3, 0.15, 0.05])
    torch.testing.assert_close(top_p_filter(probabilities, 1.0), probabilities)


def test_top_p_never_removes_the_most_likely_token():
    """Even a top_p below the top token's own mass must leave it selectable."""
    probabilities = torch.tensor([0.9, 0.07, 0.03])
    filtered = top_p_filter(probabilities, 0.01)
    assert float(filtered[0]) == 1.0
    assert float(filtered[1]) == 0.0


def test_top_p_sampling_only_ever_returns_kept_tokens():
    logits = logits_from([0.5, 0.3, 0.15, 0.05])
    params = SamplingParams(temperature=1.0, top_p=0.6)
    generator = make_generator(0)
    drawn = {int(sample(logits, params, generator=generator)) for _ in range(200)}
    assert drawn <= {0, 1}, f"top-p leaked a filtered token: {drawn}"


# -- temperature -----------------------------------------------------------

def test_low_temperature_concentrates_on_the_argmax():
    logits = logits_from([0.5, 0.3, 0.2])
    cold = SamplingParams(temperature=0.05)
    generator = make_generator(0)
    drawn = [int(sample(logits, cold, generator=generator)) for _ in range(100)]
    assert drawn.count(0) > 95, "a near-zero temperature should be nearly greedy"


def test_high_temperature_spreads_the_distribution():
    logits = logits_from([0.9, 0.07, 0.03])
    hot = SamplingParams(temperature=5.0)
    generator = make_generator(0)
    drawn = {int(sample(logits, hot, generator=generator)) for _ in range(200)}
    assert len(drawn) == 3, "a high temperature should reach the whole vocabulary"


# -- seeding ---------------------------------------------------------------


def test_the_same_seed_reproduces_the_same_draws():
    logits = logits_from([0.4, 0.35, 0.25])
    params = SamplingParams(temperature=1.0)
    first = [int(sample(logits, params, generator=make_generator(1234))) for _ in range(20)]
    second = [int(sample(logits, params, generator=make_generator(1234))) for _ in range(20)]
    assert first == second


def test_different_seeds_diverge():
    logits = logits_from([0.4, 0.35, 0.25])
    params = SamplingParams(temperature=1.0)
    first = [int(sample(logits, params, generator=make_generator(1))) for _ in range(40)]
    second = [int(sample(logits, params, generator=make_generator(2))) for _ in range(40)]
    assert first != second


def test_no_seed_gives_no_generator():
    assert make_generator(None) is None


# -- repetition penalty ----------------------------------------------------


def test_penalty_is_off_at_one():
    logits = torch.tensor([1.0, -1.0, 2.0])
    seen = torch.tensor([0, 2])
    torch.testing.assert_close(apply_repetition_penalty(logits, seen, 1.0), logits)


def test_penalty_lowers_positive_and_negative_logits_alike():
    """The sign split is the point: a penalty must never raise a seen token."""
    logits = torch.tensor([2.0, -2.0, 3.0])
    seen = torch.tensor([0, 1])
    penalised = apply_repetition_penalty(logits, seen, 2.0)

    assert float(penalised[0]) == 1.0, "positive logit should be divided"
    assert float(penalised[1]) == -4.0, "negative logit should be multiplied, i.e. pushed down"
    assert float(penalised[2]) == 3.0, "unseen token untouched"
    assert penalised[0] < logits[0] and penalised[1] < logits[1]


def test_penalty_can_change_the_greedy_choice():
    logits = torch.tensor([5.0, 4.0])
    params = SamplingParams(repetition_penalty=2.0)
    assert sample(logits, SamplingParams()).tolist() == [0]
    assert sample(logits, params, previous_tokens=torch.tensor([0])).tolist() == [1]


def test_penalty_deduplicates_repeated_history():
    """Seeing a token twice must not penalise it twice."""
    logits = torch.tensor([4.0, 1.0])
    once = apply_repetition_penalty(logits, torch.tensor([0]), 2.0)
    twice = apply_repetition_penalty(logits, torch.tensor([0, 0, 0]), 2.0)
    torch.testing.assert_close(once, twice)


def test_empty_history_is_a_no_op():
    logits = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(
        apply_repetition_penalty(logits, torch.tensor([], dtype=torch.long), 2.0), logits
    )


# -- stop conditions -------------------------------------------------------


def test_stop_state_reports_length_when_max_tokens_is_reached():
    state = StopState(SamplingParams(max_tokens=3))
    assert not state.observe(10)
    assert not state.observe(11)
    assert state.observe(12)
    assert state.finish_reason == "length"
    assert state.finished


def test_stop_state_reports_stop_on_an_eos_token():
    state = StopState(SamplingParams(max_tokens=100, stop_token_ids={7}))
    assert not state.observe(1)
    assert state.observe(7)
    assert state.finish_reason == "stop"


def test_eos_wins_over_length_on_the_final_token():
    """Hitting both at once is a stop, not a truncation — the model chose to end."""
    state = StopState(SamplingParams(max_tokens=2, stop_token_ids={9}))
    state.observe(1)
    assert state.observe(9)
    assert state.finish_reason == "stop"


def test_unfinished_state_has_no_reason():
    state = StopState(SamplingParams(max_tokens=5))
    state.observe(1)
    assert state.finish_reason is None and not state.finished


# -- shapes ----------------------------------------------------------------


def test_sample_rejects_bad_shapes():
    with pytest.raises(ValueError, match="expected"):
        sample(torch.zeros(2, 3, 4), SamplingParams())


def test_sample_returns_one_token_per_row():
    logits = torch.randn(5, 32)
    assert sample(logits, SamplingParams()).shape == (5,)
    assert sample(logits, SamplingParams(temperature=1.0)).shape == (5,)
