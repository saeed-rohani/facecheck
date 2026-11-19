import numpy as np
import pytest

from facecheck.identity import DEFAULT_THRESHOLD, decide, mean_embedding, normalize, rank, similarity

rng = np.random.default_rng(7)
ALICE, BOB, CAROL = (normalize(rng.normal(size=128)) for _ in range(3))


def sample(person, noise=0.03):
    """Another photo of the same person: the same embedding plus a little noise."""
    return normalize(person + rng.normal(scale=noise, size=128))


def test_similarity_of_same_and_different_people():
    assert similarity(ALICE, ALICE) == pytest.approx(1.0)
    assert similarity(ALICE, sample(ALICE)) > 0.8
    assert abs(similarity(ALICE, BOB)) < 0.3  # unrelated directions in 128 dimensions


def test_normalize_rejects_empty_vectors():
    with pytest.raises(ValueError):
        normalize(np.zeros(128))
    with pytest.raises(ValueError):
        normalize([np.nan] * 128)


def test_mean_embedding_is_unit_length_and_close_to_each_sample():
    samples = [sample(ALICE) for _ in range(5)]
    mean = mean_embedding(samples)
    assert np.linalg.norm(mean) == pytest.approx(1.0, abs=1e-5)
    assert all(similarity(mean, s) > 0.9 for s in samples)


def test_rank_uses_each_persons_best_sample():
    gallery = {"Alice": np.vstack([BOB, sample(ALICE)]), "Bob": np.vstack([sample(BOB), sample(BOB)])}
    ranking = rank(sample(ALICE), gallery)
    assert ranking[0][0] == "Alice"
    assert ranking[0][1] > 0.8
    assert [name for name, _ in ranking] == ["Alice", "Bob"]


GALLERY = {"Alice": np.vstack([sample(ALICE) for _ in range(3)]), "Bob": np.vstack([sample(BOB)])}


def test_identify_finds_the_right_person():
    result = decide(sample(BOB), GALLERY)
    assert result["mode"] == "identify"
    assert result["matched"] is True
    assert result["name"] == "Bob"
    assert result["threshold"] == DEFAULT_THRESHOLD
    assert result["candidates"][0]["name"] == "Bob"


def test_identify_rejects_a_stranger():
    result = decide(sample(CAROL), GALLERY)
    assert result["matched"] is False
    assert result["name"] is None
    assert result["similarity"] < DEFAULT_THRESHOLD


def test_verify_checks_only_the_claimed_person():
    assert decide(sample(ALICE), GALLERY, claimed="alice")["matched"] is True  # any letter case
    wrong = decide(sample(ALICE), GALLERY, claimed="Bob")
    assert wrong["mode"] == "verify"
    assert wrong["matched"] is False
    assert wrong["claimed_name"] == "Bob"
    with pytest.raises(KeyError):
        decide(sample(ALICE), GALLERY, claimed="Nobody")


def test_threshold_is_respected():
    query = sample(ALICE)
    score = decide(query, GALLERY)["similarity"]
    assert 0.8 < score < 0.999
    assert decide(query, GALLERY, threshold=score - 0.001)["matched"] is True
    assert decide(query, GALLERY, threshold=score + 0.001)["matched"] is False


def test_empty_gallery_is_an_error():
    with pytest.raises(ValueError):
        decide(ALICE, {})
