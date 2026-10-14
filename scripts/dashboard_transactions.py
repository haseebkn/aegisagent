"""Build a bounded investigator queue from actual development-window model alerts."""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from scripts.config import RAW_DATA_DIR
from scripts.inference_engine import run_inference


@dataclass
class AlertQueue:
    transactions: list[dict]
    scored_rows: int
    alert_count: int


def load_alert_queue(db_path, models, *, raw_data_dir=RAW_DATA_DIR, limit=200,
                     batch_size=50000):
    """Scan the complete development window and retain its highest scoring alerts.

    Labels never determine queue eligibility. Description CSV paths are resolved at
    runtime, since dbt views saved on Windows may contain unusable container paths.
    Memory used for model inputs is bounded by batch_size; retained results by limit.
    """
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Queue limit must be an integer between 1 and 1000")
    if type(batch_size) is not int or not 1 <= batch_size <= 100000:
        raise ValueError("Batch size must be an integer between 1 and 100000")
    database = Path(db_path)
    descriptions = Path(raw_data_dir) / "fraudTest.csv"
    if not database.is_file() or not descriptions.is_file():
        raise FileNotFoundError("Dashboard requires the feature database and fraudTest.csv")
    retained = pd.DataFrame()
    scored_rows = alert_count = 0
    connection = duckdb.connect(str(database), read_only=True)
    try:
        cursor = connection.execute("""
            SELECT f.* EXCLUDE (is_fraud), t.merchant, t.category
            FROM fct_fraud_features f
            LEFT JOIN (
                SELECT trans_num, merchant, category
                FROM read_csv_auto(?, types = {'trans_num': 'VARCHAR'})
            ) t ON f.trans_num = t.trans_num
            WHERE f.evaluation_role = 'development_holdout'
            ORDER BY f.trans_date_trans_time, f.trans_num
        """, [str(descriptions)])
        columns = [column[0] for column in cursor.description]
        while rows := cursor.fetchmany(batch_size):
            frame = pd.DataFrame.from_records(rows, columns=columns)
            probabilities, _, _, _, triggered = run_inference(frame, *models)
            scored_rows += len(frame)
            alert_count += int(triggered.sum())
            if triggered.any():
                alerts = frame.loc[triggered].copy()
                alerts["alert_score"] = probabilities[triggered]
                retained = pd.concat([retained, alerts], ignore_index=True).sort_values(
                    ["alert_score", "trans_date_trans_time", "trans_num"],
                    ascending=[False, True, True], kind="stable",
                ).head(limit)
    finally:
        connection.close()
    if not scored_rows:
        raise ValueError("The database has no development-holdout transactions")
    return AlertQueue(retained.to_dict(orient="records"), scored_rows, alert_count)


def data_fingerprint(db_path, raw_data_dir=RAW_DATA_DIR):
    """Invalidate the dashboard cache when its database or descriptions change."""
    paths = [Path(db_path), Path(raw_data_dir) / "fraudTest.csv"]
    return tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                 for path in paths)
