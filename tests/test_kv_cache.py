"""CPU unit tests for the KV cache's indexing and lifecycle.

These run without a GPU and without the kernel, on small float32 tensors. That is
the point: the cache's slot bookkeeping, position arithmetic and head expansion
are ordinary index math, and index math is exactly the kind of code that is cheap
to test exhaustively and expensive to debug through a CUDA kernel. Every bug
these catch would otherwise have surfaced as garbled generated text.

`gathered` is checked against an independent reference built with plain Python
loops, so a shared misunderstanding of the layout cannot pass both.
"""

from __future__ import annotations

import pytest
import torch

from engine.kv_cache import KVCache, KVCacheError

LAYERS = 3
KV_HEADS = 2
MAX_BATCH = 4
MAX_SEQ = 16
HEAD_DIM = 8


def make_cache(**overrides) -> KVCache:
    kwargs = dict(
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        max_batch=MAX_BATCH,
        max_seq=MAX_SEQ,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
    )
    kwargs.update(overrides)
    return KVCache(**kwargs)


def token_kv(marker: float, n: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinctive (H_kv, n, d) key/value pair so misplacement is visible."""
    base = torch.arange(KV_HEADS * n * HEAD_DIM, dtype=torch.float32).view(
        KV_HEADS, n, HEAD_DIM
    )
    return base + marker, base - marker


def reference_gathered(
    cache: KVCache, layer: int, slots: list[int], num_query_heads: int, pending: int
):
    """Independent reimplementation of `gathered` using explicit Python loops."""
    repeat = num_query_heads // cache.num_kv_heads
    lengths = [int(cache.seq_lens[s]) + pending for s in slots]
    max_len = max(lengths)

    keys = torch.zeros(len(slots), num_query_heads, max_len, cache.head_dim)
    values = torch.zeros_like(keys)
    for row, slot in enumerate(slots):
        for query_head in range(num_query_heads):
            kv_head = query_head // repeat
            for position in range(max_len):
                keys[row, query_head, position] = cache.keys[layer][slot, kv_head, position]
                values[row, query_head, position] = cache.values[layer][slot, kv_head, position]
    return keys, values, torch.tensor(lengths, dtype=torch.int32), max_len


# -- lifecycle -------------------------------------------------------------


def test_allocate_hands_out_distinct_slots_until_full():
    cache = make_cache()
    slots = [cache.allocate() for _ in range(MAX_BATCH)]
    assert sorted(slots) == list(range(MAX_BATCH))
    assert cache.free_slots == 0
    with pytest.raises(KVCacheError, match="no free cache slot"):
        cache.allocate()


def test_free_returns_the_slot_and_resets_its_length():
    cache = make_cache()
    slot = cache.allocate()
    key, value = token_kv(1.0, n=5)
    for layer in range(LAYERS):
        cache.append(layer, slot, key, value)
    cache.advance(slot, 5)
    assert cache.length(slot) == 5

    cache.free(slot)
    assert cache.free_slots == MAX_BATCH
    reused = cache.allocate()
    assert cache.length(reused) == 0


def test_double_free_is_an_error():
    cache = make_cache()
    slot = cache.allocate()
    cache.free(slot)
    with pytest.raises(KVCacheError, match="not allocated"):
        cache.free(slot)


def test_operations_on_an_unallocated_slot_are_errors():
    cache = make_cache()
    key, value = token_kv(1.0)
    with pytest.raises(KVCacheError, match="not allocated"):
        cache.append(0, 0, key, value)
    with pytest.raises(KVCacheError, match="not allocated"):
        cache.advance(0)


def test_slots_are_reused_oldest_first():
    """Free lists FIFO, so a just-freed row is the last one handed out again.

    Not merely incidental: reusing the most recently freed row first would hand a
    new request the physical memory a finished one may still have in flight on the
    CUDA stream. FIFO maximises the gap.
    """
    cache = make_cache()
    slots = [cache.allocate() for _ in range(MAX_BATCH)]
    cache.free(slots[0])
    cache.free(slots[1])
    assert cache.allocate() == slots[0]
    assert cache.allocate() == slots[1]


def test_freed_slot_data_is_unreachable_not_zeroed():
    """Stale bytes below a new occupant's length must never be readable."""
    cache = make_cache(max_batch=1)
    first = cache.allocate()
    key, value = token_kv(99.0, n=4)
    cache.append(0, first, key, value)
    cache.advance(first, 4)
    cache.free(first)

    second = cache.allocate()
    assert second == first, "max_batch=1 must reuse the only physical row"
    assert cache.length(second) == 0

    fresh_key, fresh_value = token_kv(1.0)
    cache.append(0, second, fresh_key, fresh_value)
    keys, _, lengths, max_len = cache.gathered(0, [second], KV_HEADS, pending=1)
    assert max_len == 1 and int(lengths[0]) == 1
    torch.testing.assert_close(keys[0, :, 0], fresh_key[:, 0])


def test_reset_frees_everything():
    cache = make_cache()
    for _ in range(3):
        cache.allocate()
    cache.reset()
    assert cache.free_slots == MAX_BATCH
    assert cache.live_slots == []


# -- writes and position arithmetic ----------------------------------------


def test_append_writes_at_the_current_length_and_advance_commits_it():
    cache = make_cache()
    slot = cache.allocate()

    first_key, first_value = token_kv(1.0, n=3)
    cache.append(0, slot, first_key, first_value)
    assert cache.length(slot) == 0, "append must not move the length by itself"
    cache.advance(slot, 3)

    second_key, second_value = token_kv(2.0, n=1)
    cache.append(0, slot, second_key, second_value)
    cache.advance(slot, 1)
    assert cache.length(slot) == 4

    torch.testing.assert_close(cache.keys[0][slot, :, 0:3], first_key)
    torch.testing.assert_close(cache.keys[0][slot, :, 3:4], second_key)
    torch.testing.assert_close(cache.values[0][slot, :, 0:3], first_value)
    torch.testing.assert_close(cache.values[0][slot, :, 3:4], second_value)


def test_layers_are_independent_but_share_one_length():
    cache = make_cache()
    slot = cache.allocate()
    for layer in range(LAYERS):
        key, value = token_kv(float(layer + 1), n=2)
        cache.append(layer, slot, key, value)
    cache.advance(slot, 2)

    assert cache.length(slot) == 2
    for layer in range(LAYERS):
        expected_key, _ = token_kv(float(layer + 1), n=2)
        torch.testing.assert_close(cache.keys[layer][slot, :, 0:2], expected_key)


def test_sequences_in_different_slots_do_not_interfere():
    cache = make_cache()
    a, b = cache.allocate(), cache.allocate()
    key_a, value_a = token_kv(1.0, n=5)
    key_b, value_b = token_kv(2.0, n=2)
    cache.append(0, a, key_a, value_a)
    cache.append(0, b, key_b, value_b)
    cache.advance(a, 5)
    cache.advance(b, 2)

    assert (cache.length(a), cache.length(b)) == (5, 2)
    torch.testing.assert_close(cache.keys[0][a, :, 0:5], key_a)
    torch.testing.assert_close(cache.keys[0][b, :, 0:2], key_b)


def test_overflow_past_max_seq_is_refused():
    cache = make_cache()
    slot = cache.allocate()
    key, value = token_kv(1.0, n=MAX_SEQ)
    cache.append(0, slot, key, value)
    cache.advance(slot, MAX_SEQ)

    extra_key, extra_value = token_kv(2.0)
    with pytest.raises(KVCacheError, match="past the cache's max_seq"):
        cache.append(0, slot, extra_key, extra_value)
    with pytest.raises(KVCacheError, match="past the cache's max_seq"):
        cache.advance(slot, 1)


def test_append_rejects_mismatched_shapes():
    cache = make_cache()
    slot = cache.allocate()
    key, value = token_kv(1.0, n=2)
    with pytest.raises(ValueError, match="shape mismatch"):
        cache.append(0, slot, key, value[:, :1])
    with pytest.raises(ValueError, match="expected key of shape"):
        cache.append(0, slot, key[0], value[0])


# -- gathered reads --------------------------------------------------------


@pytest.mark.parametrize("pending", [0, 1])
def test_gathered_matches_the_reference_implementation(pending):
    cache = make_cache()
    slots = [cache.allocate() for _ in range(3)]
    for index, slot in enumerate(slots):
        n = 2 + 3 * index  # 2, 5, 8 - deliberately ragged
        key, value = token_kv(float(index + 1), n=n)
        for layer in range(LAYERS):
            cache.append(layer, slot, key, value)
        cache.advance(slot, n)

    for layer in range(LAYERS):
        keys, values, lengths, max_len = cache.gathered(layer, slots, KV_HEADS, pending)
        ref_k, ref_v, ref_lengths, ref_max = reference_gathered(
            cache, layer, slots, KV_HEADS, pending
        )
        assert max_len == ref_max
        torch.testing.assert_close(lengths, ref_lengths)
        torch.testing.assert_close(keys, ref_k)
        torch.testing.assert_close(values, ref_v)


def test_gathered_expands_grouped_query_heads_in_the_right_order():
    """Query head h must read kv head h // repeat — adjacent copies, not tiled."""
    cache = make_cache()
    slot = cache.allocate()
    key = torch.zeros(KV_HEADS, 1, HEAD_DIM)
    key[0] = 10.0
    key[1] = 20.0
    cache.append(0, slot, key, key.clone())
    cache.advance(slot, 1)

    num_query_heads = KV_HEADS * 3
    keys, _, _, _ = cache.gathered(0, [slot], num_query_heads)
    assert keys.shape == (1, num_query_heads, 1, HEAD_DIM)
    observed = [float(keys[0, h, 0, 0]) for h in range(num_query_heads)]
    assert observed == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]

    reference_keys, _, _, _ = reference_gathered(cache, 0, [slot], num_query_heads, 0)
    torch.testing.assert_close(keys, reference_keys)


def test_gathered_respects_slot_order():
    cache = make_cache()
    slots = [cache.allocate() for _ in range(3)]
    for index, slot in enumerate(slots):
        key, value = token_kv(float(index + 1))
        cache.append(0, slot, key, value)
        cache.advance(slot, 1)

    forward, _, _, _ = cache.gathered(0, slots, KV_HEADS)
    reversed_order, _, _, _ = cache.gathered(0, list(reversed(slots)), KV_HEADS)
    torch.testing.assert_close(forward.flip(0), reversed_order)


def test_gathered_truncates_to_the_longest_sequence_in_the_batch():
    cache = make_cache()
    short, long = cache.allocate(), cache.allocate()
    for slot, n in ((short, 2), (long, 7)):
        key, value = token_kv(1.0, n=n)
        cache.append(0, slot, key, value)
        cache.advance(slot, n)

    keys, _, lengths, max_len = cache.gathered(0, [short, long], KV_HEADS)
    assert max_len == 7, "the view must cover the longest sequence, not the allocation"
    assert keys.shape[2] == 7
    assert lengths.tolist() == [2, 7]


def test_gathered_rejects_an_all_empty_batch():
    cache = make_cache()
    slot = cache.allocate()
    with pytest.raises(KVCacheError, match="nothing to attend to"):
        cache.gathered(0, [slot], KV_HEADS)


def test_gathered_rejects_a_head_count_that_is_not_a_multiple():
    cache = make_cache()
    slot = cache.allocate()
    key, value = token_kv(1.0)
    cache.append(0, slot, key, value)
    cache.advance(slot, 1)
    with pytest.raises(ValueError, match="not a multiple"):
        cache.gathered(0, [slot], KV_HEADS + 1)


def test_gathered_output_is_contiguous():
    """The kernel validates contiguity in C++; failing here is cheaper than there."""
    cache = make_cache()
    slots = [cache.allocate() for _ in range(2)]
    for slot in slots:
        key, value = token_kv(1.0, n=3)
        cache.append(0, slot, key, value)
        cache.advance(slot, 3)

    keys, values, lengths, _ = cache.gathered(0, slots, KV_HEADS * 2)
    assert keys.is_contiguous() and values.is_contiguous() and lengths.is_contiguous()


def test_memory_bytes_matches_the_allocation():
    cache = make_cache(dtype=torch.float16)
    expected = 2 * LAYERS * MAX_BATCH * KV_HEADS * MAX_SEQ * HEAD_DIM * 2
    assert cache.memory_bytes() == expected


def test_constructor_rejects_nonsense_dimensions():
    for bad in (dict(num_layers=0), dict(num_kv_heads=0), dict(max_batch=0), dict(max_seq=0)):
        with pytest.raises(ValueError):
            make_cache(**bad)
