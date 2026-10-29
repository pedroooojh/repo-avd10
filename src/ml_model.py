import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


class Classificador:
    """Prevê qual vacina a pessoa recebeu a partir do perfil (dados da SILVER)."""

    ALVO = "vacina"
    # dia_do_ano no lugar de mes: as campanhas mudam de uma semana para a outra
    # distrito_num: "DS 6" -> 6, a árvore separa grupos de distritos com uma pergunta só
    NUMERICAS = ["idade", "dose", "dia_do_ano", "distrito_num"]
    CATEGORICAS = []
    # linhas sem esses campos saem antes da divisão (sempre as mesmas, para o teste ser comparável)
    OBRIGATORIAS = ["idade", "dose", "data_vacinacao"]

    # hiperparâmetros testados com validação cruzada (só no treino)
    GRADE = {
        "max_depth": [4, 6, 8, 10, 12, 14, 16, 20, None],
        "min_samples_leaf": [1, 3, 5, 7, 10, 15, 20],
        "criterion": ["gini", "entropy"],
    }

    def __init__(self, profundidade=4, min_registros=50, numericas=None, categoricas=None, **parametros):
        self.min_registros = min_registros
        self.numericas = self.NUMERICAS if numericas is None else numericas
        self.categoricas = self.CATEGORICAS if categoricas is None else categoricas
        self.modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42, **parametros)
        self.validacao = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    def criar_colunas(self, df):
        data = pd.to_datetime(df["data_vacinacao"])
        df["ano"] = data.dt.year
        df["mes"] = data.dt.month
        df["semana"] = data.dt.isocalendar().week.astype(int)
        df["dia_do_ano"] = data.dt.dayofyear
        df["distrito_num"] = df["distrito_sanitario"].str.extract(r"(\d+)", expand=False).astype(float).fillna(0)
        return df

    def preparar(self, df):
        df = self.criar_colunas(df.copy())
        df = df.dropna(subset=self.OBRIGATORIAS + [self.ALVO])

        contagem = df[self.ALVO].value_counts()
        manter = contagem[contagem >= self.min_registros].index
        df = df[df[self.ALVO].isin(manter)]

        colunas = self.numericas + self.categoricas
        X = pd.get_dummies(df[colunas], columns=self.categoricas, dtype=int)
        y = df[self.ALVO]
        return X, y

    def dividir(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def linha_de_base(self, X_treino, y_treino, X_teste, y_teste):
        chute = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
        acuracia = accuracy_score(y_teste, chute.predict(X_teste))
        print(f"[model] linha de base: {acuracia:.1%}")

    def validar(self, X_treino, y_treino):
        """Acurácia média na validação cruzada com os parâmetros atuais (o teste não é usado)."""
        self.acuracia_validacao = cross_val_score(self.modelo, X_treino, y_treino, cv=self.validacao).mean()
        print(f"[model] acurácia na validação cruzada: {self.acuracia_validacao:.1%}")
        return self.acuracia_validacao

    def escolher_hiperparametros(self, X_treino, y_treino, grade=None):
        """Testa cada combinação da grade com validação cruzada no treino e fica com a melhor."""
        busca = GridSearchCV(DecisionTreeClassifier(random_state=42), grade or self.GRADE,
                             cv=self.validacao, n_jobs=-1)
        busca.fit(X_treino, y_treino)
        self.modelo = DecisionTreeClassifier(random_state=42, **busca.best_params_)
        self.acuracia_validacao = busca.best_score_
        print(f"[model] melhores hiperparâmetros: {busca.best_params_}")
        print(f"[model] acurácia na validação cruzada: {self.acuracia_validacao:.1%}")
        return busca.best_params_

    def treinar(self, X_treino, y_treino):
        self.modelo.fit(X_treino, y_treino)
        self.colunas = list(X_treino.columns)
        self.acuracia_treino = self.modelo.score(X_treino, y_treino)
        print(f"[model] acurácia no treino: {self.acuracia_treino:.1%}")

    def avaliar(self, X_teste, y_teste):
        previsto = self.modelo.predict(X_teste)
        self.acuracia_teste = accuracy_score(y_teste, previsto)
        self.f1_macro = f1_score(y_teste, previsto, average="macro")
        print(f"[model] acurácia no teste: {self.acuracia_teste:.1%}")
        print(classification_report(y_teste, previsto, zero_division=0))
        return previsto

    def explicar(self):
        importancia = pd.Series(self.modelo.feature_importances_, index=self.colunas)
        print("[model] colunas mais importantes:")
        print(importancia.sort_values(ascending=False).head(5).round(3))
        print(export_text(self.modelo, feature_names=self.colunas, max_depth=3))

    def matriz_confusao(self, X_teste, y_teste, arquivo="matriz_confusao.png"):
        fig, ax = plt.subplots(figsize=(11, 8))
        ConfusionMatrixDisplay.from_estimator(self.modelo, X_teste, y_teste, ax=ax, xticks_rotation=45, cmap="Oranges")
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(arquivo, dpi=150)
        print(f"[model] matriz de confusão salva em {arquivo}")
