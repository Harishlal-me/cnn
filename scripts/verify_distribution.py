import pandas as pd
df = pd.read_csv('datasets/dataset_index.csv')
print('Dataset Size:', len(df))

for task in sorted(df['task'].dropna().unique()):
    sub = df[df['task'] == task]
    print(f'\n{task.lower()}:')
    for val, count in sub['label'].value_counts().items():
        print(f'{val}: {count}')
