import pytest  # type: ignore

from interaction_protocol.utils import jaccard_modified


def test_jaccard_modified_returns_zero_for_empty_union():
    assert jaccard_modified(set(), set(), {}, 0.5) == 0.0


def test_jaccard_modified_counts_exact_then_group_matches():
    set1 = {"Value1", "Value2", "ValueA"}
    set2 = {"Value2", "Value3", "ValueB"}
    value2group = {
        "Value1": "G1",
        "Value3": "G1",
        "ValueA": "G2",
        "ValueB": "G2",
    }
    result = jaccard_modified(set1, set2, value2group, weight=0.25)
    assert result == pytest.approx(0.3)


def test_jaccard_modified_ignores_unmapped_values():
    set1 = {"Value1", "ValueX"}
    set2 = {"Value2", "ValueY"}
    value2group = {"Value1": "G1", "Value2": "G1"}
    result = jaccard_modified(set1, set2, value2group, weight=0.4)
    assert result == pytest.approx(0.1)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_jaccard_modified_rejects_invalid_weight(weight):
    with pytest.raises(ValueError):
        jaccard_modified({"Value1"}, {"Value2"}, {"Value1": "G1", "Value2": "G1"}, weight)
