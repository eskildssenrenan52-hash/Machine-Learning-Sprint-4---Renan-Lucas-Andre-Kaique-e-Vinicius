"""
Challenge Sprint 4 - Classificação linear no dataset Renewable Energy Production (2010-2020)

Como usar:
1) Baixe o CSV do Kaggle e coloque na mesma pasta deste script (ex.: renewable_energy.csv)
   https://www.kaggle.com/datasets/itsrohithere/renewable-energy-production-dataset-2010-2020
2) pip install pandas numpy scikit-learn matplotlib seaborn
3) python challenge_sprint4.py renewable_energy.csv

Saídas (pasta ./resultados): matriz de correlação, matrizes de confusão, metricas.csv, resumo.txt
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score)

TARGET = "Energy_Class"
SEED = 42
OUT = "resultados"
os.makedirs(OUT, exist_ok=True)
log = []


def say(msg=""):
    print(msg)
    log.append(str(msg))


# ---------------------------------------------------------------- 01 a)
path = sys.argv[1] if len(sys.argv) > 1 else "renewable_energy.csv"
df = pd.read_csv(path)
say(f"Dataset: {path} | formato: {df.shape}")
say(f"Colunas: {list(df.columns)}")
say(f"Nulos por coluna:\n{df.isna().sum()[df.isna().sum() > 0]}")
if TARGET not in df.columns:
    raise SystemExit(f"Coluna '{TARGET}' não encontrada. Colunas: {list(df.columns)}")
df = df.dropna().drop_duplicates().reset_index(drop=True)

# ---------------------------------------------------------------- 01 b)
# Codificação ordinal (Low < Medium < High): a ordem natural das classes torna a
# correlação interpretável (a ordem alfabética do LabelEncoder seria arbitrária).
ORDEM = ["Low", "Medium", "High"]
if set(df[TARGET].unique()) == set(ORDEM):
    mapping = {c: i for i, c in enumerate(ORDEM)}
    df[TARGET + "_enc"] = df[TARGET].map(mapping)
    le = type("Enc", (), {"classes_": np.array(ORDEM)})()
else:
    le = LabelEncoder()
    df[TARGET + "_enc"] = le.fit_transform(df[TARGET])
    mapping = {c: int(i) for i, c in enumerate(le.classes_)}
say(f"\nEncoding de {TARGET}: {mapping}")
say(f"Distribuição das classes:\n{df[TARGET].value_counts()}")

# Features: numéricas; categóricas de baixa cardinalidade viram dummies; o resto (ids/datas) é descartado
feat = df.drop(columns=[TARGET, TARGET + "_enc"])
num = feat.select_dtypes(include=np.number)
cat = feat.select_dtypes(exclude=np.number)
cat_ok = [c for c in cat.columns if cat[c].nunique() <= 10]
dropped = [c for c in cat.columns if c not in cat_ok]
X_all = pd.concat([num, pd.get_dummies(cat[cat_ok], drop_first=True).astype(int)], axis=1)
say(f"Descartadas (alta cardinalidade/ids/datas): {dropped}")
say(f"Features usadas na análise: {list(X_all.columns)}")

# ---------------------------------------------------------------- 01 c)
corr_df = X_all.copy()
corr_df[TARGET + "_enc"] = df[TARGET + "_enc"]
corr = corr_df.corr()
plt.figure(figsize=(max(7, len(corr) * 0.8), max(6, len(corr) * 0.7)))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlação (com Energy_Class codificada)")
plt.tight_layout()
plt.savefig(f"{OUT}/matriz_correlacao.png", dpi=150)
plt.close()
corr.to_csv(f"{OUT}/matriz_correlacao.csv")

# ---------------------------------------------------------------- 01 d)
tcorr = corr[TARGET + "_enc"].drop(TARGET + "_enc").abs().sort_values(ascending=False)
say(f"\n|Correlação| de cada feature com {TARGET}:\n{tcorr.round(3)}")
LIM_ALVO, LIM_COLIN = 0.10, 0.90
selected = [c for c in tcorr.index if tcorr[c] >= LIM_ALVO]
# remove colinearidade: entre pares com |r| > 0.90 mantém a mais correlacionada com o alvo
for a in list(selected):
    for b in list(selected):
        if a != b and a in selected and b in selected and abs(corr.loc[a, b]) > LIM_COLIN:
            selected.remove(b if tcorr[a] >= tcorr[b] else a)
if len(selected) == 0:
    selected = list(X_all.columns)
    say("Nenhuma feature passou o corte de correlação: usando TODAS as features.")
say(f"\nFeatures selecionadas (|r| com alvo >= {LIM_ALVO} e sem colinearidade > {LIM_COLIN}): {selected}")
say(f"Descartadas: {[c for c in X_all.columns if c not in selected]}")
say("-> Use estes números para escrever a justificativa técnica do item 01d (veja README).")

X = X_all[selected]
y = df[TARGET + "_enc"]

# Checagem extra: seleção x todas as features (validação cruzada 5-fold estratificada)
from sklearn.model_selection import StratifiedKFold, cross_val_score
_cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
_m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
acc_sel = cross_val_score(_m, X, y, cv=_cv).mean()
acc_all = cross_val_score(_m, X_all, y, cv=_cv).mean()
say(f"\nAcurácia média (CV 5-fold) - features selecionadas: {acc_sel:.4f} | todas as features: {acc_all:.4f}")

# ---------------------------------------------------------------- 02 c/d) e 03
rows = []
for nome, test_size in [("Cenario1_60-40", 0.40), ("Cenario2_85-15", 0.15)]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=SEED, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    cm = confusion_matrix(yte, pred)
    ConfusionMatrixDisplay(cm, display_labels=list(le.classes_)).plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusão - {nome} (teste={len(yte)})")
    plt.tight_layout()
    plt.savefig(f"{OUT}/matriz_confusao_{nome}.png", dpi=150)
    plt.close()

    binario = len(le.classes_) == 2
    avg = "binary" if binario else "macro"
    r = dict(cenario=nome, n_treino=len(ytr), n_teste=len(yte),
             acuracia=accuracy_score(yte, pred),
             precisao=precision_score(yte, pred, average=avg, zero_division=0),
             recall=recall_score(yte, pred, average=avg, zero_division=0))
    rows.append(r)
    say(f"\n== {nome} ==\nMatriz de confusão:\n{cm}")
    say({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    say(f"(média de precisão/recall: {avg})")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/metricas.csv", index=False)
say(f"\nComparação:\n{res.round(4).to_string(index=False)}")
with open(f"{OUT}/resumo.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
print(f"\nArquivos salvos em ./{OUT}")
