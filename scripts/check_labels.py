import pandas as pd

# Load dataset
df = pd.read_csv('emotions.csv')

print("Dataset Info:")
print(f"Total rows: {len(df)}")
print(f"\nLabel distribution:")
print(df['label'].value_counts().sort_index())

print("\n" + "="*60)
print("Sample texts for each label:")
print("="*60)

for label in sorted(df['label'].unique()):
    print(f"\n### LABEL {label} ###")
    samples = df[df['label'] == label]['text'].head(5).tolist()
    for i, text in enumerate(samples, 1):
        print(f"{i}. {text[:100]}...")
