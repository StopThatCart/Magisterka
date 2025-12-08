import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("wyniki.csv", sep=';')

base_models = ["DecisionTree", "AdaBoost", "RandomForest", "HistGradientBoosting"]
def shorten_model_names(model):
    if model.startswith(base_models[3]):
        return model.replace(base_models[3], "HGB")
    elif model.startswith(base_models[2]):
        return model.replace(base_models[2], "RF")
    elif model.startswith(base_models[1]):
        return model.replace(base_models[1], "Ada")
    elif model.startswith(base_models[0]):
        return model.replace(base_models[0], "DTree")
    else:
        return model


datasets = ["ALL", "BTU", "HEC", "HFF", "SPS", "SSH"]
metrics = ["accuracy", "balanced_accuracy", "f1_score", "precision", "recall", "stdev"]

for ds in datasets:
    heatmap_data = pd.DataFrame()

    for m in metrics:
        col = f"{ds}_{m}"
        if col not in df.columns:
            continue
        # filtrujemy wartości -1 i NaN
        df_tmp = df[df[col].notna() & (df[col] != -1)].sort_values(col, ascending=False).head(10)
        for idx, row in df_tmp.iterrows():
            model_name = shorten_model_names(row["model"])
            if model_name not in heatmap_data.index:
                heatmap_data.loc[model_name] = [None]*len(metrics)
            heatmap_data.at[model_name, m] = row[col]

    plt.figure(figsize=(10, max(6, len(heatmap_data)*0.5)))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="Blues", cbar_kws={'label': 'Wartość metryki'})
    plt.title(f"Porównanie top 10 modeli dla datasetu {ds}")
    plt.ylabel("Model")
    plt.xlabel("Metryka")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()