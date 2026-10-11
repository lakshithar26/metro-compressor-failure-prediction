"""
Data replay simulator for metro compressor sensor data.

Replays historical sensor readings at configurable speed and sends them
to the scoring API for real-time anomaly detection.
"""

import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# Default paths
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "dataset" / "metropt+3+dataset" / "MetroPT3(AirCompressor).csv"
DEFAULT_API_URL = "http://localhost:8000/predict"

# Signals to stream
SIGNALS = [
    'TP2', 'TP3', 'H1', 'DV_pressure', 'Oil_temperature', 'Motor_current',
    'COMP', 'DV_electric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses'
]


def load_data(data_path: Path, start_idx: int = 0, end_idx: int = None) -> pd.DataFrame:
    """Load sensor data from CSV."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if end_idx:
        df = df.iloc[start_idx:end_idx]
    elif start_idx > 0:
        df = df.iloc[start_idx:]

    print(f"Loaded {len(df)} records from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df


def stream_data(df: pd.DataFrame, api_url: str, speed: float = 1.0, verbose: bool = False):
    """
    Stream sensor data to the API.

    Args:
        df: DataFrame with sensor readings
        api_url: URL of the scoring API
        speed: Playback speed multiplier (1.0 = real-time, 10.0 = 10x faster)
        verbose: Print detailed output
    """
    delay = 1.0 / speed  # Time between readings
    total_rows = len(df)
    anomaly_count = 0

    print(f"\nStreaming {total_rows} readings at {speed}x speed...")
    print(f"Target API: {api_url}")
    print("-" * 60)

    start_time = time.time()

    for idx, row in df.iterrows():
        # Prepare reading payload
        reading = {
            'timestamp': row['timestamp'].isoformat(),
            **{signal: float(row[signal]) for signal in SIGNALS}
        }

        try:
            # Send to API
            response = requests.post(api_url, json=reading, timeout=5)

            if response.status_code == 200:
                result = response.json()

                if result.get('is_anomaly', False):
                    anomaly_count += 1
                    print(f"\n[ANOMALY] {row['timestamp']} - Score: {result.get('anomaly_score', 'N/A'):.4f}")
                    print(f"          Risk: {result.get('failure_risk', 'N/A')}")

                elif verbose:
                    print(f"[OK] {row['timestamp']} - Score: {result.get('anomaly_score', 'N/A'):.4f}")

            else:
                print(f"[ERROR] API returned {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to API at {api_url}")
            print("Make sure the scoring service is running: python -m src.api.scoring_service")
            return

        except Exception as e:
            print(f"[ERROR] {e}")

        # Progress update every 100 readings
        row_num = idx - df.index[0] + 1
        if row_num % 100 == 0:
            elapsed = time.time() - start_time
            rate = row_num / elapsed
            eta = (total_rows - row_num) / rate if rate > 0 else 0
            print(f"Progress: {row_num}/{total_rows} ({100*row_num/total_rows:.1f}%) - "
                  f"Rate: {rate:.1f} rows/sec - ETA: {eta:.0f}s - Anomalies: {anomaly_count}")

        # Wait for next reading
        time.sleep(delay)

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Streaming complete. {total_rows} readings in {elapsed:.1f}s")
    print(f"Total anomalies detected: {anomaly_count} ({100*anomaly_count/total_rows:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Replay metro compressor sensor data")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to CSV data file")
    parser.add_argument("--api", type=str, default=DEFAULT_API_URL,
                        help="Scoring API URL")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Playback speed multiplier (default: 10x)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start row index")
    parser.add_argument("--end", type=int, default=None,
                        help="End row index")
    parser.add_argument("--failure", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Replay specific failure period (1-4)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Failure period indices (approximate)
    FAILURE_PERIODS = {
        1: (200000, 210000),   # Failure 1: Apr 13-14
        2: (330000, 340000),   # Failure 2: Apr 28-29
        3: (1100000, 1110000), # Failure 3: Jun 23
        4: (1480000, 1516948), # Failure 4: Aug 14-17
    }

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    start_idx = args.start
    end_idx = args.end

    if args.failure:
        start_idx, end_idx = FAILURE_PERIODS[args.failure]
        print(f"Replaying Failure #{args.failure} period")

    df = load_data(data_path, start_idx, end_idx)

    # Stream to API
    stream_data(df, args.api, args.speed, args.verbose)


if __name__ == "__main__":
    main()
