# ETL – Perfil das Pessoas Vacinadas (Covid-19) – Recife

Projeto didático de ETL com Programação Orientada a Objetos (POO) e arquitetura em camadas (Medalhão).

Cada classe tem um papel: **Extract lê**, **Transform limpa**, **Load grava**.

```
API Recife ──Extract──► Load ──► BRONZE (MongoDB Atlas)
                                    │
                                 Extract ──► Transform ──► Load ──► SILVER (NeonDB)
```

- **Fonte:** https://dados.recife.pe.gov.br/dataset/perfil-das-pessoas-vacinadas-covid-19
- **Bronze:** dado cru, exatamente como veio da API (JSON → documento no Mongo) + data da carga.
- **Silver:** dado limpo, tipado e padronizado, em tabela relacional no Postgres.

> A base completa tem ~1,4 milhão de registros (338 MB). Para caber na aula e no plano gratuito
> do Atlas, extraímos só uma amostra (`LIMITE_REGISTROS` no `src/config.py`, padrão 5000).

## Estrutura

```
ProjDataScience/
├── src/
│   ├── config.py      # configurações (URL da API, nomes do banco/tabela, limite)
│   ├── extract.py     # class Extract   -> lê da API e da bronze
│   ├── transform.py   # class Transform -> limpa com pandas
│   └── load.py        # class Load      -> grava na bronze e na silver
├── main.py            # roda as 3 etapas
├── requirements.txt
└── .env.example       # modelo do .env com as credenciais
```

O `main.py` inteiro:

```python
from src.extract import Extract
from src.transform import Transform
from src.load import Load

extract = Extract()
transform = Transform()
load = Load()

# API -> BRONZE (MongoDB Atlas)
registros = extract.ler_api()
load.salvar_bronze(registros)

# BRONZE -> SILVER (NeonDB)
df = extract.ler_bronze()
df = transform.limpar(df)
load.salvar_silver(df)
```

## Preparação (antes da aula)

1. **MongoDB Atlas** (https://cloud.mongodb.com)
   - Crie um cluster gratuito (M0).
   - Em *Database Access*, crie um usuário e senha.
   - Em *Network Access*, libere seu IP (ou `0.0.0.0/0` para a aula).
   - Em *Connect → Drivers*, copie a string `mongodb+srv://...`.
2. **NeonDB** (https://neon.tech)
   - Crie um projeto gratuito.
   - Em *Connection string*, copie a URL `postgresql://...?sslmode=require`.

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
copy .env.example .env          # depois edite o .env com suas credenciais
python main.py
```

Saída esperada:

```
[extract] 500 registros baixados da API
...
[extract] 5000 registros baixados da API
[load] 5000 documentos gravados na bronze
[extract] 5000 documentos lidos da bronze
[transform] ~4900 linhas prontas para a silver
[load] ~4900 linhas gravadas na tabela 'silver_vacinados'
```

## O que a transformação faz (bronze → silver)

| Problema na bronze                                   | Tratamento na silver                                  |
|------------------------------------------------------|-------------------------------------------------------|
| Colunas técnicas (`_id`, `_data_carga`)              | Não são lidas (projeção no `find` do Mongo)           |
| Textos com espaços sobrando e caixa misturada        | `strip()` + `upper()`                                 |
| `categoria` vazia (`""`)                             | Vira nulo (`NULL`)                                    |
| `data_vacinacao` como texto `2022-05-09T00:00:00`    | Convertida para `DATE`                                |
| `descricao_dose`                                     | Renomeada para `dose` e convertida para inteiro       |
| `vacina_fabricante` = `"3 - COMIRNATY (PFIZER)"`     | Nova coluna `vacina` = `"COMIRNATY (PFIZER)"`         |
| `cnes` = `"DS 4: CNES: 0000639 - UBS JOAQUIM ..."`   | Novas colunas `distrito_sanitario` e `unidade_saude`  |
| Linhas sem idade/data e linhas duplicadas            | Removidas                                             |

## Conferindo o resultado

**No Atlas** (aba *Collections*): banco `vacinacao` → coleção `bronze_vacinados`.

**No Neon** (aba *SQL Editor*):

```sql
-- vacinados por vacina
SELECT vacina, COUNT(*) AS total
FROM silver_vacinados
GROUP BY vacina
ORDER BY total DESC;

-- vacinados por distrito sanitário e sexo
SELECT distrito_sanitario, sexo, COUNT(*) AS total
FROM silver_vacinados
GROUP BY distrito_sanitario, sexo
ORDER BY distrito_sanitario;
```

## Conceitos de POO trabalhados

- **Classe e objeto:** cada etapa do ETL é uma classe; o `main.py` cria um objeto de cada.
- **Responsabilidade única:** `Extract` só lê, `Transform` só limpa, `Load` só grava.
- **Métodos:** uma mesma classe pode ter várias ações do mesmo tipo (`Load.salvar_bronze` e `Load.salvar_silver`).
- **Encapsulamento:** o `main.py` não sabe como o Mongo ou o Postgres funcionam, só chama os métodos.

## Desafios para os alunos

1. Adicionar a coluna `faixa_etaria_inicio` (inteiro) a partir de `"75 A 79 ANOS"`.
2. Criar uma classe `Gold` (arquivo `src/gold.py`) que grave uma tabela agregada (ex.: total por vacina e dose).
3. Fazer a carga da bronze ser incremental (só inserir `_id`s que ainda não existem).
