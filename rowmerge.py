import logging
import re
from typing import Iterable, List, Tuple
from argparse import ArgumentParser

import pandas as pd


# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def discover_mappings(columns: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Given a sequence of column names, this function attempts to find mappings between them.
    Only id columns are considered: those ending in ".id" or ".@id".

    A mapping is detected when there are two columns of the form "bar.id" and "foo.bar.id".
    If "bar.id" could be mapped to multiple targets ("foo.bar.id", "baz.bar.id"),
    only the first such mapping is included in the output.

    Returns a list of tuples of the form:
    [("w.x.y.id", "y.id"), ("y.z.id", "z.id"), ...]
    """
    # find columns which appear to be top-level indexes (i.e. potential targets for links)
    index_cols = set()
    for col in columns:
        # match "foo.id" or "bar.@id". Don't match "foo.bar.id" or "foo.id.bar".
        if re.match(r"\w+\.@?id$", col.strip()) is not None:
            index_cols.add(col)
    # find other columns which appear to match with those indexes
    mappings = []
    for col in columns:
        # position of the second to last "." (or -1 if it isn't found)
        offset = col.rfind(".", 0, col.rfind("."))
        # suffix contains everything after the second to last "." (or the whole string if not found)
        suffix = col[offset+1:]
        if offset > 0 and suffix in index_cols:
            mappings.append((col, suffix))
            # once we've found one mapping for a given index, stop searching for more
            # there's no satisfactory way to merge rows on many-to-many relationships anyway
            index_cols.remove(suffix)
    return mappings


def do_joins(df_in: pd.DataFrame, mappings: Iterable[Tuple[str, str]]) -> pd.DataFrame:
    """
    :param df_in: The DataFrame containing all the data which should be joined.
    :param mappings: A sequence of pairs of column names to join
    :return: A new DataFrame with appropriate rows merged.
    """
    df_out = df_in
    for ref_col, id_col in mappings:
        # copy all the rows which have a value (i.e. not NaN) in the ID column
        mask = df_out.loc[:, id_col].notna()
        df_to_join = df_out[mask].copy()

        # if no rows have values then we can safely skip joining
        if len(df_to_join) == 0:
            continue

        # remove those rows from the main table
        # we also switch to a temporary working copy here in case we need to rollback changes
        df_tmp = df_out.drop(index=df_to_join.index)

        # remove any columns which are entirely NaN from both tables
        # this should (hopefully) result in two tables which don't share any columns
        df_to_join.dropna(axis='columns', inplace=True, how='all')
        df_tmp.dropna(axis='columns', inplace=True, how='all')
        # if there are overlapping columns then we can't safely perform the join on this particular index
        # the pandas merge function would produce an error, and even if it didn't we'd risk overwriting data
        overlap = set(df_to_join.columns).intersection(set(df_tmp.columns))
        if len(overlap) > 0:
            logger.warning(f"Encountered overlapping columns when trying to merge {id_col}:\n"
                           f"\t{', '.join(overlap)}")
            continue

        # Now we do the actual join operation.
        logger.info(f"Joining {id_col} onto {ref_col}")
        df_out = pd.merge(df_tmp, df_to_join, left_on=ref_col, right_on=id_col, how='outer')
    return df_out


def merge_rows(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Performs row merging as described in:
    https://gitlab.com/hestia-earth/hestia-schema/-/snippets/2194329
    The input should be a DataFrame with different data nodes on different rows,
    linked together by columns named *.id or *.@id.
    Returns a new DataFrame with related nodes merged onto the same row.
    """
    # Remove any rows which are totally empty
    df_out = df_in.dropna(how='all')
    # If all rows are empty then we can just return immediately
    if len(df_out) == 0:
        return df_out

    col_mappings = discover_mappings(df_out.columns)
    if len(col_mappings) == 0:
        logger.warning("Failed to find any join operations which could be performed. "
                       "Maybe you didn't include any index columns (*.id or *.@id) in the data export?")
        return df_out

    df_out = do_joins(df_out, col_mappings)

    # It's possible there were empty columns dropped which should appear in the output
    for col in set(df_in.columns).difference(df_out.columns):
        df_out.insert(0, col, float("NaN"))

    # reorder the columns to match the input csv
    # this is important because in the web UI the user can request the columns to be in a particular order
    df_out = df_out[df_in.columns]
    return df_out


def transform_csv(input_path_or_buf, output_path_or_buf):
    df_in = pd.read_csv(input_path_or_buf, na_values=['-'])
    df_out = merge_rows(df_in)
    df_out.to_csv(output_path_or_buf, index=False, na_rep="-")


if __name__ == '__main__':
    parser = ArgumentParser(description="Merge related rows in CSV files from the hestia.earth platform.")
    parser.add_argument('input', type=str, help="the CSV file to process")
    parser.add_argument('output', type=str, help="filename where the output CSV will be written")
    args = parser.parse_args()
    transform_csv(args.input, args.output)
