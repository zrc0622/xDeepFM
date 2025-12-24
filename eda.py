import os
import pandas as pd
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

THEME_COLOR = '#660974'

sns.set_style("whitegrid")

sns.set_context("talk", font_scale=1.5)

plt.rcParams['axes.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['figure.titlesize'] = 34

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_eda(config_path, output_dir="eda"):
    cfg = load_config(config_path)
    data_cfg = cfg['data']
    file_path = data_cfg['path']
    
    eda_rows =  5000000 # 45840617
    print(f"Loading raw data for EDA (First {eda_rows} rows)")
    
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    col_names = ['label'] + dense_features + sparse_features
    
    try:
        data = pd.read_csv(file_path, sep='\t', header=None, names=col_names, nrows=eda_rows)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    stats_log = []

    # 1. 分析标签分布 (Label Distribution)
    print("Analyzing Label Distribution")
    label_counts = data['label'].value_counts(normalize=True).sort_index()
    # 绘制饼图
    stats_log.append(f"Label Distribution:\n{label_counts.to_string()}\n")
    
    plt.figure(figsize=(10, 10))
    colors = ['#E0E0E0', THEME_COLOR] 

    plt.pie(label_counts, labels=['Not Click (0)', 'Click (1)'], autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 24})
    # plt.title('CTR Label Distribution', fontsize=32, color=THEME_COLOR, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
    plt.close()

    # 2. 分析缺失值 (Missing Values)
    print("Analyzing Missing Values")
    missing = data.isnull().sum()
    # 计算缺失百分比并绘制条形图
    missing = missing[missing > 0]
    missing_percent = (missing / len(data)) * 100
    missing_percent = missing_percent.sort_values(ascending=False)
    
    if not missing_percent.empty:
        stats_log.append(f"Top 10 Features with Missing Values:\n{missing_percent.head(10).to_string()}\n")
        
        plt.figure(figsize=(16, 10)) # 画布调大
        
        df_missing = pd.DataFrame({
            'Feature': missing_percent.index,
            'Missing_Percent': missing_percent.values
        })
        
        sns.barplot(data=df_missing, x='Feature', y='Missing_Percent', color=THEME_COLOR)
        plt.xticks(rotation=90) 
        # plt.title('Percentage of Missing Values per Feature', color=THEME_COLOR, fontweight='bold')
        plt.ylabel('Missing Percentage (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values.png'))
        plt.close()
    else:
        stats_log.append("No missing values found (or already filled).\n")

    # 3. 分析数值特征 (Dense Features)
    print("Analyzing Dense Features")
    # 计算统计量 (mean, std, min, max)
    dense_stats = data[dense_features].describe().T[['mean', 'std', 'min', '50%', 'max']]
    stats_log.append(f"Dense Features Statistics:\n{dense_stats.to_string()}\n")

    # 计算相关性矩阵并绘制热力图 (Heatmap)
    corr_matrix = data[['label'] + dense_features].corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Purples", vmin=0, vmax=1, annot_kws={"size": 18})
    # plt.title('Correlation Matrix (Dense Features & Label)', color=THEME_COLOR, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_dense.png'))
    plt.close()

    plt.figure(figsize=(18, 10))

    # 绘制对数变换后的箱线图 (Boxplot)，观察离群点和分布
    df_dense = data[dense_features].fillna(0)
    df_dense = df_dense.clip(lower=0)
    data_log = np.log1p(df_dense)
    
    sns.boxplot(data=data_log, color=THEME_COLOR)
    # plt.title('Distribution of Dense Features (Log Scale)', color=THEME_COLOR, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Log(Value + 1)')
    plt.savefig(os.path.join(output_dir, 'dense_boxplot_log.png'))
    plt.close()

    # 4. 分析稀疏特征基数 (Cardinality)    
    print("Analyzing Sparse Features Cardinality")
    # 统计每个类别特征有多少个唯一取值 (Vocabulary Size)
    cardinality = data[sparse_features].nunique().sort_values(ascending=False)
    # 绘制条形图，使用对数坐标轴，因为不同特征的基数差异可能巨大
    stats_log.append(f"Top 5 High Cardinality Sparse Features:\n{cardinality.head(5).to_string()}\n")
    stats_log.append(f"Total Unique Categories (sum of vocab): {cardinality.sum()}\n")

    plt.figure(figsize=(16, 10))
    
    df_cardinality = pd.DataFrame({
        'Feature': cardinality.index,
        'Unique_Values': cardinality.values
    })
    
    sns.barplot(data=df_cardinality, x='Feature', y='Unique_Values', color=THEME_COLOR)
    plt.yscale('log')
    plt.xticks(rotation=90)
    # plt.title('Cardinality of Sparse Features (Log Scale)', color=THEME_COLOR, fontweight='bold')
    plt.ylabel('Unique Values Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sparse_cardinality.png'))
    plt.close()

    report_path = os.path.join(output_dir, 'eda_report_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(stats_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xdeepfm.yaml")
    args = parser.parse_args()
    
    run_eda(args.config)