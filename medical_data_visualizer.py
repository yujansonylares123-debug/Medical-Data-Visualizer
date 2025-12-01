import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# 1. Importar datos
df = pd.read_csv("medical_examination.csv")

# 2. Añadir columna 'overweight'
# BMI = weight(kg) / (height(m))^2
bmi = df["weight"] / ((df["height"] / 100) ** 2)
df["overweight"] = (bmi > 25).astype(int)

# 3. Normalizar datos: 0 = bueno, 1 = malo
# colesterol y gluc: 1 -> 0 (normal), >1 -> 1 (malo)
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


def draw_cat_plot():
    # 4. Crear DataFrame para el gráfico categórico

    # 4.1 Convertir a formato largo con pd.melt
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
    )

    # 4.2 Agrupar por cardio, variable y valor, y contar
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"], as_index=False)
        .size()
        .rename(columns={"size": "total"})
    )

    # 4.3 Crear gráfico categórico con seaborn.catplot
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
    ).fig

    # No modificar estas dos líneas
    fig.savefig("catplot.png")
    return fig


def draw_heat_map():
    # 5. Limpiar datos para el heatmap

    # Condiciones:
    # - ap_lo <= ap_hi
    # - height entre percentil 2.5 y 97.5
    # - weight entre percentil 2.5 y 97.5
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 6. Matriz de correlación
    corr = df_heat.corr()

    # 7. Máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 8. Figura de matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 9. Heatmap con seaborn
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    # No modificar estas dos líneas
    fig.savefig("heatmap.png")
    return fig