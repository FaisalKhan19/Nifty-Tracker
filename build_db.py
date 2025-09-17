import os
import re
import pandas as pd
import argparse
import sqlite3

parser = argparse.ArgumentParser()

DB_PATH = "NSE_Weekly.db"
# Define folder where all CSV files are stored
data_folder = f"./ohlc_weekly_data/"

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create the 'symbols' table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS symbols (
        symbol TEXT PRIMARY KEY,
        name TEXT
    )
""")
conn.commit()

# Get a list of all CSV files in the folder
csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".csv")])

# Loop through all CSV files and append data
for file in csv_files:
    file_path = os.path.join(data_folder, file)
    base_name = os.path.basename(file_path)
    name_no_ext, _ = os.path.splitext(base_name)
    # Remove known suffix like _weekly
    if name_no_ext.endswith("_weekly"):
        name_no_ext = name_no_ext[: -len("_weekly")]
    # Remove exchange suffix like .NSE that may be part of the base name
    symbol = name_no_ext.replace(".NSE", "")
    safe_symbol = re.sub(r"[^A-Za-z0-9_]", "_", symbol)
    # Add symbol to 'symbols' table if not already present
    cursor.execute("INSERT OR IGNORE INTO symbols (symbol, name) VALUES (?, ?)", (symbol.replace("USDT", ""), symbol))
    conn.commit()

    # Define the table schema dynamically
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS "{safe_symbol}" (
        date TEXT PRIMARY KEY,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adjusted_close REAL,
        volume REAL
    )
    """
    cursor.execute(create_table_query)
    conn.commit()
    # Read CSV
    # Read CSV using header row, select needed columns only
    df = pd.read_csv(
        file_path,
        header=0,
        usecols=[
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ],
    )

    # Remove symbol column; we store per-symbol tables per file
    df = df.drop(columns=["symbol"], errors="ignore")

    # Drop invalid timestamps
    df = df.dropna(subset=['date'])

    # Normalize date to ISO format (string) for SQLite TEXT PK
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Filter out rows that already exist in the table
    try:
        existing_dates = pd.read_sql(f"SELECT date FROM \"{safe_symbol}\"", conn)
    except Exception:
        existing_dates = pd.DataFrame(columns=["date"])  # table just created
    if not existing_dates.empty:
        df = df[~df["date"].isin(existing_dates["date"])].copy()

    # Sort by date
    df = df.sort_values("date")

    # Insert into SQLite database
    if not df.empty:
        df.to_sql(safe_symbol, conn, if_exists="append", index=False)

    print(f"Data for {symbol} inserted into {DB_PATH}")

# Close connection
conn.close()
