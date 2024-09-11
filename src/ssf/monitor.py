import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def getMonitor():
    """
    Args    : None
    
    Return  : None
    """
    path = "~/data/modelfish/data.parquet"
    rpath = os.path.expanduser(path)
    if not os.path.isdir(os.path.dirname(rpath)):
        print("""
        디렉토리가 존재하지 않습니다.
        """)
        exit()

    if not os.path.isfile(rpath):
        print("""
        학습한 데이터가 존재하지 않습니다.
        """)
        exit()

    df = pd.read_parquet(rpath)
    df['length'] = df['length'].astype(float)
    df['weight'] = df['weight'].astype(float)
    df['target'] = df['target'].astype(int)
    
    sns.scatterplot(data=df, x='length', y='weight', hue='target')
    plt.title('Selfstudy Model')
    plt.show()
