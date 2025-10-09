import pandas as pd

def _merge_with_prefix(base_df: pd.DataFrame, indicator_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Concatenate indicator columns to base_df with a prefix, aligned by index."""
    if indicator_df is None or len(indicator_df) == 0:
        return base_df
    base_df['date'] = base_df.index
    aligned_base = base_df.reset_index(drop=True)
    aligned_ind = indicator_df.reset_index(drop=True).add_prefix(f"{prefix}_")
    merged = pd.concat([aligned_base, aligned_ind], axis=1)
    merged.index = pd.to_datetime(merged['date'])
    return merged
