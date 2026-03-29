import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def split_train_test(x, y, train_size=0.8, random_state=42):
    train_set_size = int(len(x) * train_size)
    
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(x))
        
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:]
    
    return x.iloc[train_indices], x.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

def z_score(x):
    return (x - x.mean()) / x.std(ddof=1)

class CancerDataHelper():
    def load_dataset(one_hot=True, normalize=True):
        cancer_data = pd.read_csv("./input/breast_cancer.csv")
        cancer_data = cancer_data.drop(columns=["id"])
        
        if (one_hot):
            cancer_data = pd.get_dummies(cancer_data, columns=["diagnosis"], drop_first=True, dtype=int)
        
        x_train, x_test, y_train, y_test = split_train_test(cancer_data.drop("diagnosis_M", axis=1), cancer_data["diagnosis_M"])
        
        if (normalize):
            x_train = z_score(x_train)
            x_test = z_score(x_test)
        
        return x_train, x_test, y_train, y_test

    def plot_outcome_distribution(x, y):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        malignant_counts = y.value_counts()
        labels = ["Benign", "Malignant"]
        colors = ["#2ecc71", "#e74c3c"]

        ax.pie(malignant_counts.values, labels=labels, autopct="%1.1f%%", 
            colors=colors, startangle=90, textprops={"fontsize": 12})
        ax.set_title("Class Distribution in Training Data", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

        print(f"Benign: {malignant_counts[0]} ({malignant_counts[0]/len(y)*100:.1f}%)")
        print(f"Malignant: {malignant_counts[1]} ({malignant_counts[1]/len(y)*100:.1f}%)")
        
    def plot_correlation(x, y):
        fig, ax = plt.subplots(figsize=(20, 16))

        correlation_data = x.copy()
        correlation_data["diagnosis_M"] = y
        correlation_matrix = correlation_data.corr()

        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    center=0, square=True, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.show()
        
    def plot_boxplots(x, y):
        fig, axes = plt.subplots(6, 5, figsize=(24, 20))
        axes = axes.ravel()

        viz_data = x.copy()
        viz_data["Outcome"] = y.replace({1: "Malignant", 0: "Benign"})

        for idx, feature in enumerate(x.columns):
            sns.boxplot(data=viz_data, x="Outcome", y=feature, ax=axes[idx], 
                        palette=["#2ecc71", "#e74c3c"])
            axes[idx].set_title(f"{feature} Distribution by Outcome", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Outcome", fontsize=11)
            axes[idx].set_ylabel(feature, fontsize=11)

        plt.tight_layout()
        plt.show()