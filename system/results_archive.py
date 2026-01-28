# system/results_archive.py
# Utility module for managing results archive
# Allows cumulative BO training across multiple runs

import os
import pandas as pd
from datetime import datetime
import config


def ensure_archive_dir():
    """Create the archive directory if it doesn't exist."""
    if not os.path.exists(config.RESULTS_ARCHIVE_DIR):
        os.makedirs(config.RESULTS_ARCHIVE_DIR)
        print(f"  Created results archive directory: {config.RESULTS_ARCHIVE_DIR}")


def archive_current_results(description=None):
    """
    Archive the current result.csv to the archive folder with timestamp.

    Args:
        description (str, optional): Description to include in filename

    Returns:
        str: Path to the archived file, or None if no results to archive
    """
    if not os.path.exists(config.RESULTS_CSV_FILE):
        print("  [INFO] No result.csv to archive")
        return None

    ensure_archive_dir()

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if description:
        # Clean description for filename
        clean_desc = "".join(c if c.isalnum() or c in "-_" else "_" for c in description)
        filename = f"result_{timestamp}_{clean_desc}.csv"
    else:
        filename = f"result_{timestamp}.csv"

    archive_path = os.path.join(config.RESULTS_ARCHIVE_DIR, filename)

    # Copy current results to archive
    df = pd.read_csv(config.RESULTS_CSV_FILE)
    df.to_csv(archive_path, index=False)

    print(f"  Archived {len(df)} results to: {archive_path}")
    return archive_path


def load_all_archived_results():
    """
    Load and merge all results from the archive folder.

    Returns:
        pd.DataFrame: Combined DataFrame with all archived results,
                      or empty DataFrame if no archives exist
    """
    if not os.path.exists(config.RESULTS_ARCHIVE_DIR):
        print("  [INFO] No archive directory found")
        return pd.DataFrame()

    # Find all CSV files in archive
    archive_files = [f for f in os.listdir(config.RESULTS_ARCHIVE_DIR)
                     if f.endswith('.csv')]

    if not archive_files:
        print("  [INFO] No archived results found")
        return pd.DataFrame()

    # Load and concatenate all files
    dfs = []
    total_rows = 0
    for filename in sorted(archive_files):
        filepath = os.path.join(config.RESULTS_ARCHIVE_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
            total_rows += len(df)
            print(f"    Loaded {len(df)} results from {filename}")
        except Exception as e:
            print(f"    [WARNING] Failed to load {filename}: {e}")

    if not dfs:
        return pd.DataFrame()

    # Combine all dataframes
    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates based on sim_id (keep first occurrence)
    # This handles cases where same results were archived multiple times
    if 'sim_id' in combined.columns:
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=['sim_id'], keep='first')
        after_dedup = len(combined)
        if before_dedup != after_dedup:
            print(f"    Removed {before_dedup - after_dedup} duplicate sim_ids")

    print(f"  Total archived results loaded: {len(combined)}")
    return combined


def load_all_results_for_bo():
    """
    Load all available results for BO training:
    1. Current result.csv (if exists)
    2. All archived results

    Merges and deduplicates by sim_id.

    Returns:
        pd.DataFrame: Combined DataFrame with all available results
    """
    print("\n--- Loading all results for BO ---")

    dfs = []

    # Load current results
    if os.path.exists(config.RESULTS_CSV_FILE):
        try:
            current_df = pd.read_csv(config.RESULTS_CSV_FILE)
            dfs.append(current_df)
            print(f"  Loaded {len(current_df)} results from current result.csv")
        except Exception as e:
            print(f"  [WARNING] Failed to load current results: {e}")

    # Load archived results
    archived_df = load_all_archived_results()
    if not archived_df.empty:
        dfs.append(archived_df)

    if not dfs:
        print("  [WARNING] No results found anywhere!")
        return pd.DataFrame()

    # Combine all
    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate by sim_id
    if 'sim_id' in combined.columns:
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=['sim_id'], keep='last')
        after_dedup = len(combined)
        if before_dedup != after_dedup:
            print(f"  Removed {before_dedup - after_dedup} duplicates (kept latest)")

    print(f"  Total results available for BO: {len(combined)}")
    return combined


def get_next_sim_id():
    """
    Get the next available sim_id based on all results (current + archived).

    Returns:
        int: Next available sim_id
    """
    all_results = load_all_results_for_bo()

    if all_results.empty or 'sim_id' not in all_results.columns:
        return 1

    max_id = int(all_results['sim_id'].max())
    return max_id + 1


def list_archives():
    """List all archived result files with summary info."""
    if not os.path.exists(config.RESULTS_ARCHIVE_DIR):
        print("No archive directory found")
        return

    archive_files = sorted([f for f in os.listdir(config.RESULTS_ARCHIVE_DIR)
                           if f.endswith('.csv')])

    if not archive_files:
        print("No archived results found")
        return

    print(f"\n{'='*60}")
    print("Results Archive Summary")
    print(f"{'='*60}")
    print(f"Archive location: {config.RESULTS_ARCHIVE_DIR}")
    print(f"\n{'Filename':<45} {'Rows':>8}")
    print("-" * 55)

    total_rows = 0
    for filename in archive_files:
        filepath = os.path.join(config.RESULTS_ARCHIVE_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            rows = len(df)
            total_rows += rows
            print(f"{filename:<45} {rows:>8}")
        except:
            print(f"{filename:<45} {'ERROR':>8}")

    print("-" * 55)
    print(f"{'TOTAL':<45} {total_rows:>8}")
