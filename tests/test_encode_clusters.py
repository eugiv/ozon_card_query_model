from datetime import date

import pandas as pd
import pytest

from cluster_data import rolling_training_windows
from cluster_recommender import encoding_as_of, main


@pytest.mark.parametrize("revision", ["main", "v1", "12345", "z" * 40])
def test_encoder_refuses_mutable_model_revision(tmp_path, revision):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--dictionary",
                str(tmp_path / "missing.csv"),
                "--cutoff",
                "2026-01-08",
                "--revision",
                revision,
                "--output",
                str(tmp_path / "embeddings.npz"),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "embeddings.npz").exists()


def test_full_history_embeddings_use_the_explicit_dictionary_date():
    actual = encoding_as_of(date(2026, 10, 3), 14, 1, date(2026, 6, 26))
    assert actual == pd.Timestamp("2026-06-26T23:59:59.999999999Z")


def test_legacy_encoder_date_is_unchanged():
    cutoff = date(2026, 10, 2)
    first = rolling_training_windows(cutoff, 14, 3)[0]
    assert encoding_as_of(cutoff, 14, 3).date() == first.history_end


def test_future_dictionary_fails_before_reading_or_loading_the_model(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--dictionary", str(tmp_path / "missing.csv"), "--cutoff", "2026-10-03",
              "--dictionary-as-of", "2026-10-04", "--revision", "a" * 40,
              "--output", str(tmp_path / "embeddings.npz")])
    assert error.value.code == 2
    assert "позже cutoff" in capsys.readouterr().err
    assert not (tmp_path / "embeddings.npz").exists()
