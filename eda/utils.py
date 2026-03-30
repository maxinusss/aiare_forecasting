from functools import reduce
import pandas as pd 

def merge_dataframes_on_keys(dfs, keys, how='outer'):
    """Merge a list of dataframes on a set of keys.

    Args:
        dfs (list[pd.DataFrame]): DataFrames to merge.
        keys (list[str]): Keys to merge on.
        how (str): Merge style (inner, outer, left, right).

    Returns:
        pd.DataFrame: Merged result.
    """
    if not isinstance(dfs, list) or len(dfs) == 0:
        raise ValueError("dfs must be a non-empty list of DataFrames")
    if not isinstance(keys, list) or len(keys) == 0:
        raise ValueError("keys must be a non-empty list of column names")

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=keys, how=how), dfs)
    return merged_df