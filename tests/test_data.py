import pandas as pd
import pytest

from interaction_protocol.data import load_debate_data, save_debate_data


def test_parquet_round_trip_restores_nested_debate_types(tmp_path):
    """Parquet round trips restore lists and per-round value sets."""
    dataframe = pd.DataFrame(
        {
            "n_rounds": [2],
            "final_verdict": ["NTA"],
            "Agent_1_verdicts": [["YTA", "NTA"]],
            "Agent_1_messages": [["First", "Second"]],
            "Agent_1_values": [[{"Honesty", "Empathy"}, {"Fairness"}]],
        }
    )
    path = tmp_path / "debates.parquet"

    save_debate_data(dataframe, path)
    restored = load_debate_data(path)

    pd.testing.assert_frame_equal(restored, dataframe)
    assert isinstance(restored.loc[0, "Agent_1_verdicts"], list)
    assert isinstance(restored.loc[0, "Agent_1_values"], list)
    assert isinstance(restored.loc[0, "Agent_1_values"][0], set)


def test_pickle_loader_preserves_legacy_data(tmp_path):
    """The loader remains compatible with trusted legacy pickle files."""
    dataframe = pd.DataFrame({"n_rounds": [1], "Agent_1_verdicts": [["NTA"]]})
    path = tmp_path / "debates.pkl"
    dataframe.to_pickle(path)

    restored = load_debate_data(path)

    pd.testing.assert_frame_equal(restored, dataframe)


def test_unsupported_format_raises(tmp_path):
    """Unsupported debate-data extensions fail with a clear error."""
    with pytest.raises(ValueError, match="Unsupported debate data format"):
        load_debate_data(tmp_path / "debates.csv")
