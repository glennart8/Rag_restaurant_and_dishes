import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
from restaurant_data import RAW_RESTAURANT_DATA

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DB_PATH = "restaurants_and_food_db"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

df = pd.DataFrame(RAW_RESTAURANT_DATA)

# Vektorisera beskrivningstexten
print("Skapar vektorer...")
df["menu_text"] = df["menu"].apply(lambda m: " ".join(m.get("dishes", []) + m.get("drinks", [])))
df["vector"] = df["menu_text"].apply(lambda x: embedding_model.encode(x).tolist())

# Skapa LanceDB-tabell
db = lancedb.connect(DB_PATH)
table = db.create_table(
    "restaurants_db",
    data=df,
    mode="overwrite"
)


print("Databasen 'restaurants_db' är skapad och uppdaterad.")
