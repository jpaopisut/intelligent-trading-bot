import pytest
import numpy.testing as npt

from common.utils import *
from common.gen_signals import *
from common.gen_labels_topbot import *

def test_extremum_labels():
    data = [10, 30, 50, 70, 90, 70, 50, 30, 9]
    data = [10, 40, 30, 70, 90, 50, 60, 30, 9]
    sr = pd.Series(data)
    sr = pd.Series(data * 2)
    level_frac = 0.5
    tolerance_frac = 0.1

    maximums = find_all_extremums(sr, True, level_frac, tolerance_frac)
    minimums = find_all_extremums(sr, False, level_frac, tolerance_frac)

    # Merge into a sequence of interleaving minimums and maximums
    all = list()
    all.extend(maximums)
    all.extend(minimums)

    all.sort(key=lambda x: x[1])

    # We have indexes (x coordinates) and need to find their y values
    extr_x = [x[1] for x in all]
    extr_y = [sr[x] for x in extr_x]
    extr_df = pd.DataFrame({'x': extr_x, 'y': extr_y})

    # Plotting with seaborn was here for manual/visual debugging only (seaborn is an
    # optional dependency, see requirements.txt) and is not part of the actual test.
    # The test itself only needs to verify that extremum detection runs without errors
    # and produces a non-empty result.
    assert len(extr_x) > 0
    assert len(extr_df) == len(extr_x)

def test_interval_and_aggregation():
    data = [10, 40, 30, 70, 90, 50, 60, 30, 9]
    sr = pd.Series(data * 2)

    df = pd.DataFrame(data={'close': sr, 'score': sr / 10})

    level_frac = 0.5
    tolerance_frac = 0.1

    # For debugging
    maximums = find_all_extremums(sr, True, level_frac, tolerance_frac)

    # Add label
    df, _ = add_extremum_features(df, column_name='close', level_fracs=[level_frac], tolerance_frac=tolerance_frac, out_names=['is_close_top'])

    # Aggregate score with chosen parameters.
    # `aggregate_scores` was deprecated and removed (see commit 0a2f3c4), replaced by
    # `generate_smoothen_scores` which uses a config dict instead of positional parameters.
    smoothen_config = {"columns": ["score"], "window": 2, "names": "score_agg"}
    df, _ = generate_smoothen_scores(df, smoothen_config)

    threshold = 6
    interval_df = find_interval_precision(df, label_column='is_close_top', score_column='score_agg', threshold=threshold)

    pass
