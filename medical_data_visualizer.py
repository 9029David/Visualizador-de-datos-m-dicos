import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
peso = df.weight
altura = df.height / 100
df['overweight'] = peso / altura**2
df['overweight'] = df['overweight'].map(lambda x: 1 if x > 25 else 0)

# 3
df['gluc'] = df['gluc'].map(lambda x: 1 if x > 1 else 0)
df['cholesterol'] = df['cholesterol'].map(lambda x: 1 if x > 1 else 0)

# 4
def draw_cat_plot():
    # 5
    index = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    df_cat = pd.melt(df, id_vars=['id','cardio'], value_vars=index, var_name='variable', value_name='value')

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable','value']).size().reset_index(name='total')
    
    # 7
    g = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar', height=5, aspect=1.2)
    plt.show()

    # 8
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975)) &
        (df['ap_lo'] <= df['ap_hi'])
        ]
        
    # 12
    corr = df_heat.corr().round(1)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15
    sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=ax, fmt='.1f')

    # 16
    fig.savefig('heatmap.png')
    return fig

