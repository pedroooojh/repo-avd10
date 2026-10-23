# Exercício em grupo – Melhorando o classificador de vacinas

## Contexto

Depois do ETL (API → bronze → silver), usamos os dados da **SILVER** para treinar um modelo
de Machine Learning. O `train.py` usa a classe `Classificador` (`src/ml_model.py`) para
**prever qual vacina a pessoa recebeu** a partir do perfil dela (idade, dose, data, sexo,
raça/cor e distrito sanitário).

Rodando `python train.py` hoje, o resultado é:

```
[model] linha de base: 42.8%       <- "chutar" sempre a vacina mais comum
[model] acurácia no teste: 63.2%   <- árvore de decisão com profundidade 4
```

O modelo já é melhor que o chute, mas ainda erra mais de 1 em cada 3 pessoas. Olhando o
relatório, algumas vacinas quase nunca são acertadas (veja o `recall` da CORONAVAC e da
ASTRAZENECA).

## Objetivo

Em grupos, **melhorem o modelo até passar de 80% de acurácia no conjunto de teste.**

E sim, é possível — usando só a árvore de decisão.

## Regras

Para que os resultados dos grupos sejam comparáveis:

1. **Mesmos dados:** não alterem `LIMITE_REGISTROS` nem o ETL. Usem a silver como está.
2. **Mesma divisão treino/teste:** mantenham o método `dividir` como está
   (`test_size=0.2`, `random_state=42`, `stratify=y`).
3. **O teste é sagrado:** não usem o conjunto de teste para escolher hiperparâmetros.
   Para comparar configurações, usem só o treino (ex.: validação cruzada com `cross_val_score`
   ou `GridSearchCV`).
4. **Proibido usar a coluna `lote`.** Discutam no grupo o porquê (ver perguntas abaixo).
5. **Mantenham a organização em classe:** as melhorias devem ficar dentro do `Classificador`
   (ou de uma classe nova), e o `train.py` continua sendo só a "receita" que chama os métodos.
6. **Só árvore de decisão:** o modelo continua sendo o `DecisionTreeClassifier`. Vocês podem
   ajustar os parâmetros dele (ex.: `max_depth`, `min_samples_leaf`), mas não podem usar outros
   modelos.

## Por onde começar (dicas)

Não precisam seguir todas — escolham, testem e **anotem o resultado de cada tentativa**.

1. **Entendam o modelo atual.** Leiam a saída de `explicar()` e a imagem
   `matriz_confusao.png`. Quais colunas a árvore mais usa? Quais vacinas ela confunde entre si?
2. **Olhem para o tempo.** As vacinas usadas mudaram ao longo de 2022, e campanhas mudam de uma
   semana para a outra. Será que `mes` é a melhor forma de representar a data?
3. **Olhem para a silver.** Ela tem colunas que o modelo ainda não usa (`grupo`, `categoria`,
   `unidade_saude`, `faixa_etaria`...). Quais delas fazem sentido para o problema?
4. **Ajustem a árvore.** Testem hiperparâmetros como `max_depth`, `min_samples_leaf` e
   `criterion`. Profundidade maior sempre é melhor? Comparem a acurácia no treino e no teste.

## Perguntas para discutir no grupo

1. Com a coluna `lote` o modelo passa de 90%. Por que isso **não** é um bom modelo?
   (Pesquisem o termo *data leakage* / vazamento de dados.)
2. Acurácia de 80% significa que o modelo é bom para **todas** as vacinas? Olhem o
   `classification_report`: o que aconteceu com as classes pequenas?
3. Uma árvore sem limite de profundidade acerta quase 100% no treino. Por que isso não se repete
   no teste?
4. Das mudanças que vocês fizeram, qual teve **maior impacto**? Por quê?

## Entrega

1. **Código** com as alterações (`src/ml_model.py`, `train.py` e o que mais criarem).
2. **Saída do `python train.py`** com a acurácia final e a nova `matriz_confusao.png`.
3. **Tabela de experimentos** (pode ser num `RESULTADOS.md`), uma linha por tentativa:

   | # | O que mudou | Acurácia (validação) | Acurácia (teste) |
   |---|-------------|----------------------|------------------|
   | 0 | Modelo original | – | 63,2% |
   | 1 | ... | ... | ... |

4. **Respostas** às perguntas para discussão (curtas, 2–4 linhas cada).
5. **Apresentação em sala** do resultado: o que tentaram, o que funcionou e a acurácia final.

## Critérios de avaliação

| Critério | Peso |
|----------|------|
| Acurácia no teste acima de 80%, respeitando as regras | 40% |
| Tabela de experimentos (mostra o caminho, não só o resultado final) | 25% |
| Respostas às perguntas para discussão | 20% |
| Organização do código (classes, métodos com responsabilidade clara) | 15% |

## Desafios bônus

- Melhorar o `f1-score` médio (`macro avg`) — e não só a acurácia.
- Rodar o modelo com uma amostra maior da base (em uma cópia separada) e comparar: mais dados
  ajudam?
