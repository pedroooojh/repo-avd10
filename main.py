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
