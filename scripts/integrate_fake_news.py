"""
Integrate FakeNewsNet into dataset_index.csv  (memory-safe version)
====================================================================
"""
import os, hashlib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS   = os.path.join(ROOT, "datasets")

MAX_PER_FILE = 5000   # cap per CSV to avoid OOM


def load_safe(path, label_val, max_rows=MAX_PER_FILE):
    """Read FakeNewsNet CSV, extract title/text, return standardised rows."""
    df = pd.read_csv(path, nrows=max_rows, on_bad_lines="skip")
    text_col = None
    for c in ("title", "text", "content", "headline"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        print(f"  SKIP {os.path.basename(path)}: no text column")
        return pd.DataFrame()

    out = pd.DataFrame({
        "dataset_name": os.path.basename(path).replace(".csv", ""),
        "text":         df[text_col].astype(str),
        "image_path":   "",
        "label":        label_val,
        "task":         "fake_news",
    })
    return out


def main():
    frames = []
    sources = [
        ("gossipcop_fake.csv",              1, 3000),
        ("gossipcop_real.csv",              0, 3000),
        ("politifact_fake.csv",             1, 500),
        ("politifact_real.csv",             0, 500),
        ("BuzzFeed_fake_news_content.csv",  1, 500),
        ("BuzzFeed_real_news_content.csv",  0, 500),
    ]

    for fn, lbl, cap in sources:
        p = os.path.join(DS, fn)
        if os.path.isfile(p):
            df = load_safe(p, lbl, cap)
            if len(df):
                print(f"  {fn}: {len(df)} rows (label={lbl})")
                frames.append(df)
        else:
            print(f"  {fn}: NOT FOUND")

    if not frames:
        print("ERROR: No fake news data found.")
        return

    fake_df = pd.concat(frames, ignore_index=True)

    # Clean
    fake_df = fake_df[fake_df["text"].str.strip() != ""]
    fake_df = fake_df[fake_df["text"].str.lower() != "nan"]
    fake_df = fake_df.dropna(subset=["text"])

    # Deduplicate
    fake_df["_h"] = fake_df["text"].apply(
        lambda t: hashlib.md5(t.strip().lower().encode()).hexdigest()
    )
    before = len(fake_df)
    fake_df = fake_df.drop_duplicates(subset="_h").drop(columns=["_h"])
    print(f"\n  Dedup: {before} -> {len(fake_df)}")

    # Append to dataset_index.csv
    idx = os.path.join(DS, "dataset_index.csv")
    existing = pd.read_csv(idx)

    # Remove any old fake_news rows to avoid duplicates on re-run
    existing = existing[existing["task"] != "fake_news"]

    for col in existing.columns:
        if col not in fake_df.columns:
            fake_df[col] = ""
    fake_df = fake_df[existing.columns]

    combined = pd.concat([existing, fake_df], ignore_index=True)
    combined.to_csv(idx, index=False)

    print(f"\n  Existing (no fake): {len(existing)}")
    print(f"  Fake news added:   {len(fake_df)}")
    print(f"  New total:         {len(combined)}")
    print(f"\n  Task distribution:")
    print(combined["task"].value_counts().to_string())

    fake_labels = fake_df["label"].value_counts()
    print(f"\n  Fake news label split:")
    print(f"    real (0): {fake_labels.get(0, 0)}")
    print(f"    fake (1): {fake_labels.get(1, 0)}")


if __name__ == "__main__":
    main()
