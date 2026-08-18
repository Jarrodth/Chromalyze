import numpy as np
import pytest

from chromalyze.stems import combine_stems


def test_combine_stems_sums_non_drum_stems():
    vocals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    bass = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    other = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    mixed = combine_stems({"vocals": vocals, "bass": bass, "other": other})

    assert np.allclose(mixed, [2.5, 3.5, 4.5])


def test_combine_stems_excludes_drums_by_default():
    vocals = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    drums = np.array([100.0, 100.0, 100.0], dtype=np.float32)

    mixed = combine_stems({"vocals": vocals, "drums": drums})

    assert np.allclose(mixed, vocals)


def test_combine_stems_attenuates_drums_when_given_a_gain():
    vocals = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    drums = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    mixed = combine_stems({"vocals": vocals, "drums": drums}, drum_attenuation=0.1)

    assert np.allclose(mixed, [2.0, 2.0, 2.0])


def test_combine_stems_full_attenuation_matches_a_plain_sum():
    vocals = np.array([1.0, 2.0], dtype=np.float32)
    drums = np.array([3.0, 4.0], dtype=np.float32)

    mixed = combine_stems({"vocals": vocals, "drums": drums}, drum_attenuation=1.0)

    assert np.allclose(mixed, [4.0, 6.0])


def test_combine_stems_trims_to_shortest_stem():
    vocals = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    bass = np.array([1.0, 1.0], dtype=np.float32)

    mixed = combine_stems({"vocals": vocals, "bass": bass})

    assert len(mixed) == 2
    assert np.allclose(mixed, [2.0, 2.0])


def test_combine_stems_requires_at_least_one_stem():
    with pytest.raises(ValueError):
        combine_stems({})
