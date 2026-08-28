"""Generate a small fraudTrain/fraudTest pair so CI can exercise the real dbt
pipeline without the 500 MB Kaggle download.

The fixture reproduces the shape that matters: the same columns, a chronological
train/test boundary, repeated cards and merchants so velocity and target encodings
have something to aggregate, and a realistic class imbalance.
"""
import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

COLUMNS = ["column00", "trans_date_trans_time", "cc_num", "merchant", "category",
           "amt", "first", "last", "gender", "street", "city", "state", "zip",
           "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
           "merch_lat", "merch_long", "is_fraud"]

CATEGORIES = ["grocery_pos", "shopping_net", "misc_net", "gas_transport",
              "entertainment", "health_fitness", "travel", "kids_pets"]
STATES = ["NC", "MA", "IL", "CA", "NY", "TX"]
DEFAULT_TRAIN_ROWS = 12000
DEFAULT_TEST_ROWS = 2500


def generate(rows, start, cards, merchants, fraud_rate, rng, row_offset=0):
    out = []
    t = start
    for i in range(rows):
        card = cards[rng.randrange(len(cards))]
        merchant = merchants[rng.randrange(len(merchants))]
        # Advance irregularly so trailing-window velocity varies across rows.
        t += timedelta(minutes=rng.randint(1, 240))
        is_fraud = 1 if rng.random() < fraud_rate else 0
        amt = round(rng.uniform(500, 1200) if is_fraud else rng.uniform(1, 150), 2)
        state = STATES[card % len(STATES)]
        lat, lon = 35 + (card % 10), -80 - (card % 10)
        out.append({
            "column00": row_offset + i,
            "trans_date_trans_time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": 4000000000000000 + card,
            "merchant": f"fraud_Merchant_{merchant}",
            "category": CATEGORIES[merchant % len(CATEGORIES)],
            "amt": amt,
            "first": f"First{card}", "last": f"Last{card}",
            "gender": "F" if card % 2 else "M",
            "street": f"{card} Test Street", "city": f"City{card % 20}",
            "state": state, "zip": 10000 + card,
            "lat": lat, "long": lon, "city_pop": 1000 * (card % 50 + 1),
            "job": "Tester", "dob": "1985-01-01",
            "trans_num": f"{'t' if row_offset == 0 else 'e'}{i:08d}",
            "unix_time": int(t.timestamp()),
            "merch_lat": round(lat + rng.uniform(-0.5, 0.5), 6),
            "merch_long": round(lon + rng.uniform(-0.5, 0.5), 6),
            "is_fraud": is_fraud,
        })
    return out


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    # A long reference window is intentional. With only 6,000 rows, the first 90
    # days of card-history warm-up made up enough of the training distribution to
    # create a false-positive card_txn_cnt drift alert (PSI 0.354) even though the
    # feature pipeline was correct. At 12,000 rows it reaches a representative
    # steady state while keeping CI small (PSI 0.055 on the fixed seed).
    ap.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    ap.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cards = list(range(1, 61))
    merchants = list(range(1, 25))

    train = generate(args.train_rows, datetime(2019, 1, 1), cards, merchants,
                     0.02, rng)
    last = datetime.strptime(train[-1]["trans_date_trans_time"], "%Y-%m-%d %H:%M:%S")
    test = generate(args.test_rows, last + timedelta(minutes=1), cards, merchants,
                    0.02, rng, row_offset=args.train_rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "fraudTrain.csv", train)
    write(out / "fraudTest.csv", test)
    print(f"wrote {len(train):,} train + {len(test):,} test rows to {out.resolve()}")
    print(f"total rows: {len(train) + len(test)}")


if __name__ == "__main__":
    main()
