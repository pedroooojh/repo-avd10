"""Roda cada tentativa do exercício e imprime a tabela de experimentos (markdown).

A acurácia de validação vem da validação cruzada no treino; o teste só é medido no fim de cada
tentativa, para registro. A escolha da configuração final foi feita pela coluna de validação.
"""
from src.extract import Extract
from src.ml_model import Classificador

PERFIL = ["sexo", "raca_cor", "distrito_sanitario"]

# (descrição, colunas numéricas, colunas categóricas, parâmetros fixos ou None para usar a grade)
TENTATIVAS = [
    ("Modelo original (mes, max_depth=4)", ["idade", "dose", "ano", "mes"], PERFIL, {"profundidade": 4}),
    ("Original + grade de hiperparâmetros", ["idade", "dose", "ano", "mes"], PERFIL, None),
    ("Árvore sem limite (max_depth=None)", ["idade", "dose", "ano", "mes"], PERFIL, {"profundidade": None}),
    ("dia_do_ano no lugar de ano/mes (max_depth=4)", ["idade", "dose", "dia_do_ano"], PERFIL, {"profundidade": 4}),
    ("semana no lugar de ano/mes + grade", ["idade", "dose", "semana"], PERFIL, None),
    ("dia_do_ano + grade", ["idade", "dose", "dia_do_ano"], PERFIL, None),
    ("+ unidade_saude", ["idade", "dose", "dia_do_ano"], PERFIL + ["unidade_saude"], None),
    ("+ grupo", ["idade", "dose", "dia_do_ano"], PERFIL + ["grupo"], None),
    ("+ grupo, categoria, faixa_etaria, sistema_origem", ["idade", "dose", "dia_do_ano"],
     PERFIL + ["grupo", "categoria", "faixa_etaria", "sistema_origem"], None),
    ("dia_do_ano sem sexo e raca_cor", ["idade", "dose", "dia_do_ano"], ["distrito_sanitario"], None),
    ("Distrito como número (distrito_num) — FINAL", ["idade", "dose", "dia_do_ano", "distrito_num"], [], None),
    ("Final + grupo", ["idade", "dose", "dia_do_ano", "distrito_num"], ["grupo"], None),
]


def rodar(df, descricao, numericas, categoricas, parametros):
    print(f"\n=== {descricao}")
    classificador = Classificador(numericas=numericas, categoricas=categoricas, **(parametros or {}))
    X, y = classificador.preparar(df)
    X_treino, X_teste, y_treino, y_teste = classificador.dividir(X, y)

    if parametros is None:
        escolhidos = classificador.escolher_hiperparametros(X_treino, y_treino)
    else:
        classificador.validar(X_treino, y_treino)
        escolhidos = {"max_depth": classificador.modelo.max_depth}
    classificador.treinar(X_treino, y_treino)
    classificador.avaliar(X_teste, y_teste)
    return escolhidos, classificador


df = Extract().ler_silver()
linhas = []
for numero, (descricao, numericas, categoricas, parametros) in enumerate(TENTATIVAS):
    escolhidos, c = rodar(df, descricao, numericas, categoricas, parametros)
    parametros_texto = ", ".join(f"{k}={v}" for k, v in escolhidos.items())
    linhas.append(f"| {numero} | {descricao} | {parametros_texto} | {c.acuracia_treino:.1%} | "
                  f"{c.acuracia_validacao:.1%} | {c.acuracia_teste:.1%} | {c.f1_macro:.2f} |")

print("\n| # | O que mudou | Hiperparâmetros | Acurácia (treino) | Acurácia (validação) | Acurácia (teste) | F1 macro (teste) |")
print("|---|---|---|---|---|---|---|")
print("\n".join(linhas))
