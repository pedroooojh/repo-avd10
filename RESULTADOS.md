# Resultados – Melhorando o classificador de vacinas

**Acurácia final no teste: 81,7%** (validação cruzada no treino: 82,6%), só com `DecisionTreeClassifier`,
mesma silver, mesmo `dividir()` e sem a coluna `lote`.

## O que mudou no código

- `src/ml_model.py` (classe `Classificador`)
  - `criar_colunas()`: novas representações da data (`dia_do_ano`, `semana`, além de `ano`/`mes`) e
    `distrito_num` (`"DS 6"` → `6`).
  - `preparar()`: as colunas usadas vêm de `numericas`/`categoricas` (parâmetros do construtor), o que
    permite testar cada tentativa sem mexer na classe. As linhas descartadas são sempre as mesmas
    (sem `idade`, `dose` ou `data_vacinacao`), então a divisão treino/teste é idêntica à original.
  - `validar()`: acurácia média da validação cruzada (5 folds × 3 repetições, só no treino).
  - `escolher_hiperparametros()`: `GridSearchCV` sobre `max_depth`, `min_samples_leaf` e `criterion`,
    só no treino. O modelo fica com a melhor combinação.
  - `matriz_confusao()`: figura maior, para os nomes das vacinas não ficarem sobrepostos.
  - `dividir()` **não foi alterado**.
- `train.py`: continua sendo a receita; agora chama `escolher_hiperparametros()` antes de `treinar()`.
- `experimentos.py` (novo): roda todas as tentativas abaixo e imprime esta tabela.

Modelo final: colunas `idade`, `dose`, `dia_do_ano`, `distrito_num`;
`criterion="entropy"`, `max_depth=14`, `min_samples_leaf=10`.

## Tabela de experimentos

A configuração final foi escolhida pela coluna **validação** (média da validação cruzada no treino).
O teste aparece só para registro.

| # | O que mudou | Hiperparâmetros | Acurácia (treino) | Acurácia (validação) | Acurácia (teste) | F1 macro (teste) |
|---|---|---|---|---|---|---|
| 0 | Modelo original (`mes`, `max_depth=4`) | max_depth=4 | 64,6% | 63,8% | 61,8% | 0,40 |
| 1 | Original + grade de hiperparâmetros | gini, max_depth=8, min_samples_leaf=10 | 72,3% | 69,1% | 66,0% | 0,53 |
| 2 | Árvore sem limite de profundidade | max_depth=None | 87,1% | 62,7% | 58,9% | 0,47 |
| 3 | `dia_do_ano` no lugar de `ano`/`mes` (`max_depth=4`) | max_depth=4 | 71,3% | 71,7% | 69,3% | 0,52 |
| 4 | `semana` no lugar de `ano`/`mes` + grade | gini, max_depth=14, min_samples_leaf=20 | 79,2% | 76,6% | 74,9% | 0,68 |
| 5 | `dia_do_ano` + grade | entropy, max_depth=12, min_samples_leaf=10 | 85,5% | 81,3% | 79,7% | 0,71 |
| 6 | + `unidade_saude` | entropy, max_depth=12, min_samples_leaf=10 | 85,5% | 81,3% | 79,7% | 0,71 |
| 7 | + `grupo` | entropy, max_depth=12, min_samples_leaf=10 | 85,5% | 81,2% | 80,3% | 0,72 |
| 8 | + `grupo`, `categoria`, `faixa_etaria`, `sistema_origem` | entropy, max_depth=12, min_samples_leaf=10 | 85,6% | 81,3% | 79,8% | 0,72 |
| 9 | `dia_do_ano` sem `sexo` e `raca_cor` | entropy, max_depth=14, min_samples_leaf=7 | 86,5% | 82,0% | 80,7% | 0,72 |
| **10** | **Distrito como número (`distrito_num`) – FINAL** | **entropy, max_depth=14, min_samples_leaf=10** | **86,4%** | **82,6%** | **81,7%** | **0,73** |
| 11 | Final + `grupo` | entropy, max_depth=14, min_samples_leaf=10 | 86,5% | 82,4% | 81,1% | 0,72 |

Observações sobre o caminho:

- **0 → 3:** com a mesma árvore de profundidade 4, só trocar `mes` por `dia_do_ano` já sobe a validação de 63,8% para 71,7%.
- **1 vs 5:** ajustar hiperparâmetros com `mes` chega a 69,1%; com `dia_do_ano`, a 81,3%. Árvores mais
  profundas só ajudam quando a coluna de data tem detalhe para elas aproveitarem.
- **4:** `semana` fica no meio do caminho (76,6%), o que confirma que quanto mais fina a data, melhor.
- **6:** `unidade_saude` não muda nada porque, nesta amostra, cada distrito tem uma única unidade (8 e 8).
- **7, 8, 11:** `grupo`, `categoria` e `faixa_etaria` repetem o que `idade` já informa (o grupo é definido pela idade/prioridade). `categoria` é quase toda nula (4298 de 4716).
- **9:** tirar `sexo` e `raca_cor` *melhorou* a validação: são colunas que não têm relação com a vacina disponível e só dão ao modelo chance de decorar ruído.
- **10:** `distrito_num` deixa a árvore separar grupos de distritos com uma única pergunta, em vez de
  uma pergunta por distrito (one-hot). O ganho (82,0% → 82,6%) é pequeno, menor que o desvio da
  validação (~1,3 p.p.), mas foi a melhor configuração pela validação, que é o critério combinado.

Os números vêm de uma silver com 4716 linhas, montada com os mesmos 5000 registros da API
(`LIMITE_REGISTROS`) e o mesmo `Transform`. Nela a linha de base é a mesma do enunciado (42,8%), mas o modelo
original dá 61,8% em vez de 63,2%, então a amostra pode diferir um pouco da usada em sala. Para reproduzir: `python experimentos.py`.

## Saída do `python train.py`

```
[extract] 4716 linhas lidas da silver
[model] linha de base: 42.8%
[model] melhores hiperparâmetros: {'criterion': 'entropy', 'max_depth': 14, 'min_samples_leaf': 10}
[model] acurácia na validação cruzada: 82.6%
[model] acurácia no treino: 86.4%
[model] acurácia no teste: 81.7%
                                               precision    recall  f1-score   support

CHADOX1NCOV-19 - OXFORD/ASTRAZENECA (FIOCRUZ)       0.70      0.70      0.70       155
                           COMIRNATY (PFIZER)       0.88      0.87      0.87       404
              COMIRNATY (PFIZER) - PEDIÁTRICA       0.88      0.94      0.91        72
               CORONAVAC - SINOVAC (BUTANTAN)       0.73      0.53      0.62        60
    CORONAVAC - SINOVAC (BUTANTAN) PEDIÁTRICA       0.56      0.36      0.44        25
 JANSSEN COVID-19 VACCINE (JOHNSON & JOHNSON)       0.80      0.89      0.84       228

                                     accuracy                           0.82       944
                                    macro avg       0.76      0.72      0.73       944
                                 weighted avg       0.81      0.82      0.81       944

[model] colunas mais importantes:
dia_do_ano      0.524
idade           0.335
dose            0.093
distrito_num    0.048
```

A primeira pergunta da árvore é `idade <= 11.5` (crianças: Pfizer pediátrica ou Coronavac pediátrica).
Depois vêm `dia_do_ano` e `dose`: por exemplo, adultos na 3ª/4ª dose depois do dia ~47 (meados de fevereiro)
recebem muito mais Janssen.

Matriz de confusão: `matriz_confusao.png`. As confusões que sobram são AstraZeneca ↔ Pfizer/Janssen
(as três foram usadas como reforço em adultos nos mesmos períodos), Coronavac adulto → Pfizer, e
Coronavac pediátrica ↔ Pfizer pediátrica/Coronavac adulto.

## Respostas às perguntas para discussão

**1. Com `lote` o modelo passa de 90%. Por que isso não é um bom modelo?**
É *data leakage* (vazamento de dados): o lote vem impresso no frasco e cada lote pertence a uma única vacina.
Nesta amostra, só 3 dos 70 lotes aparecem com mais de uma vacina, provavelmente erro de digitação. Com `lote`, uma
árvore sem limite chega a 98,5% no teste, mas está só "lendo a resposta". Quem sabe o lote já sabe a vacina,
e lotes novos nunca vistos no treino não seriam reconhecidos. O modelo não aprende nada sobre o perfil
da pessoa nem sobre a campanha.

**2. 80% de acurácia significa que o modelo é bom para todas as vacinas?**
Não. A acurácia é dominada pelas classes grandes (Pfizer é 43% do teste). O recall da Coronavac adulta é 53%
e o da Coronavac pediátrica é 36% (acerta 9 de 25). O F1 macro (0,73) fica bem abaixo da acurácia (0,82).
Classes pequenas têm poucos exemplos para a árvore criar folhas próprias (com `min_samples_leaf=10`) e se
parecem com outras na idade e no período. Além disso, vacinas com menos de 50 registros são removidas em
`preparar()` e nem entram na avaliação.

**3. Uma árvore sem limite acerta quase tudo no treino. Por que isso não se repete no teste?**
Sem limite, a árvore cria folhas até decorar cada pessoa do treino, inclusive o ruído (tentativa 2:
87,1% no treino, 58,9% no teste, pior que a árvore de profundidade 4). Essas regras específicas não valem
para pessoas novas, o que se chama *overfitting* (sobreajuste). Aqui o treino não chega a 100% porque há
pessoas com as mesmas colunas e vacinas diferentes, que nenhuma árvore consegue separar.

**4. Qual mudança teve maior impacto? Por quê?**
Trocar `mes` por `dia_do_ano`: +8 p.p. com a mesma profundidade (0 → 3) e +12 p.p. depois do ajuste de
hiperparâmetros (1 → 5). Em fevereiro e março as quatro vacinas foram aplicadas, e qual delas a
unidade tinha mudava de uma semana para outra. Com `mes`, o mês inteiro vira um só valor. Com o dia, a
árvore corta exatamente na data em que o estoque ou a campanha mudou. O ajuste de `max_depth`/`min_samples_leaf`
veio em segundo lugar e só rendeu tanto porque a data passou a ter detalhe para ser aproveitado.
