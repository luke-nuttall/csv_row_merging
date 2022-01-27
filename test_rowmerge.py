from pathlib import Path
from io import BytesIO

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import rowmerge


@pytest.mark.parametrize("columns", [
    [],
    [""],
    [".", "..", "..."],
    ["foo.id", "barfoo.id"],
    ["a.b.c", "d.e", "f", "ostrich", "something.with.id.in.the.middle", "with.id"],
    ["漢字", "🎈.id"],  # shouldn't crash even if we have non-ASCII unicode in our headers
])
def test_discover_mappings_no_matches(columns):
    assert rowmerge.discover_mappings(columns) == []


def test_discover_mappings_at():
    columns = ["foo.@id", "bar.foo.@id", "spam.and.eggs.and.spam", "thing.@id", "some.thing.@id"]
    expected = [("bar.foo.@id", "foo.@id"), ("some.thing.@id", "thing.@id")]
    # convert the output to a set because order doesn't matter
    assert set(rowmerge.discover_mappings(columns)) == set(expected)


def test_discover_mappings_no_at():
    columns = ["foo.id", "bar.foo.id", "spam.and.eggs.and.spam", "thing.id", "some.thing.id"]
    expected = [("bar.foo.id", "foo.id"), ("some.thing.id", "thing.id")]
    # convert the output to a set because order doesn't matter
    assert set(rowmerge.discover_mappings(columns)) == set(expected)


def test_discover_mappings_multiple_matches():
    columns = ["foo.id", "bar.foo.id", "baz.foo.id"]
    expected = [("bar.foo.id", "foo.id")]
    assert rowmerge.discover_mappings(columns) == expected


def assert_frame_fuzzy_equal(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Asserts that two DataFrame objects contain the same data,
    but ignores some irrelevant differences including:
        - order of rows
        - data types (remember, we're exporting to CSV in the end anyway)
    It is assumed that both df1 and df2 have a column named "cycle.@id"
    """
    df1 = df1.set_index("cycle.@id").sort_index()
    df2 = df2.set_index("cycle.@id").sort_index()
    assert_frame_equal(df1, df2, check_dtype=False, check_index_type=False)


def assert_csv_equal(path_in: Path, path_out: Path):
    """
    Takes two paths to CSV files on disk.
    Asserts that applying `transform_csv()` to `path_in` produces a CSV with the same data as `path_out`.
    """
    buf = BytesIO()
    rowmerge.transform_csv(path_in, buf)
    buf.seek(0)
    df_out = pd.read_csv(buf, na_values=['-'])
    df_goal = pd.read_csv(path_out, na_values=['-'])
    assert_frame_fuzzy_equal(df_out, df_goal)


def test_csv_official():
    # these are the example files provided in the problem specification
    # includes examples of both one-to-one and one-to-many relationships
    path_in = Path("test_data/official_input.csv")
    path_out = Path("test_data/official_output.csv")
    assert_csv_equal(path_in, path_out)


def test_csv_column_overlap():
    # rows which are potential targets for merging but contain overlapping data
    path_in = Path("test_data/column_overlap_input.csv")
    path_out = Path("test_data/column_overlap_output.csv")
    assert_csv_equal(path_in, path_out)


def test_csv_no_joins():
    # data which doesn't include the necessary id columns for merging
    path_in = Path("test_data/no_joins_input.csv")
    path_out = Path("test_data/no_joins_output.csv")
    assert_csv_equal(path_in, path_out)


def test_csv_empty_columns():
    # all the "site.*" columns have been filled with "-"
    path_in = Path("test_data/no_site_input.csv")
    path_out = Path("test_data/no_site_output.csv")
    assert_csv_equal(path_in, path_out)


def test_csv_no_data():
    # nothing but "-" in every field
    path_in = Path("test_data/empty_input.csv")
    path_out = Path("test_data/empty_output.csv")
    assert_csv_equal(path_in, path_out)


def test_csv_all_columns_selected():
    # csv taken from the hestia.earth site with every available column included
    path_in = Path("test_data/all_columns_selected_input.csv")
    path_out = Path("test_data/all_columns_selected_output.csv")
    assert_csv_equal(path_in, path_out)
