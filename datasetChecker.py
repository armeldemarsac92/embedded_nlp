import pandas as pd
import sys

# --- CONFIGURATION ---
FILE_PATH = "DataSetTeensyv9_ULTRA_CLEAN.csv"
SAMPLES_PER_TOPIC = 20  # How many sentences to grab per topic


def extract_random_samples():
    print(f"📂 Loading {FILE_PATH}...")

    try:
        # Load data
        df = pd.read_csv(FILE_PATH)

        # Check required columns
        if 'topic' not in df.columns or 'french_sentence' not in df.columns:
            print("❌ Error: CSV must contain 'topic' and 'french_sentence' columns.")
            return

        # Get unique topics
        unique_topics = df['topic'].unique()
        print(f"✅ Found {len(unique_topics)} topics. Extracting {SAMPLES_PER_TOPIC} samples each...\n")

        print("=" * 60)
        print("👇 COPY FROM HERE DOWN TO SEND FOR ANALYSIS 👇")
        print("=" * 60)

        for topic in sorted(unique_topics):
            # Filter by topic
            topic_df = df[df['topic'] == topic]

            # Sample random rows (handle cases with fewer rows than requested)
            n = min(SAMPLES_PER_TOPIC, len(topic_df))
            if n > 0:
                samples = topic_df.sample(n=n)

                print(f"\n[{topic}]")
                for _, row in samples.iterrows():
                    # Clean up output: remove explicit quotes if pandas adds them
                    sentence = str(row['french_sentence']).strip()
                    print(f" > {sentence}")
            else:
                print(f"\n[{topic}] - ⚠️ NO DATA FOUND")

        print("\n" + "=" * 60)
        print("👆 COPY UP TO HERE 👆")
        print("=" * 60)

    except FileNotFoundError:
        print(f"❌ Error: Could not find '{FILE_PATH}'.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    extract_random_samples()