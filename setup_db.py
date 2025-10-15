import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
from restaurant_data import RAW_RESTAURANT_DATA
import json

DB_PATH = "restaurants_and_food_db"
model = SentenceTransformer('all-MiniLM-L6-v2')

# Skapa vektor
lancedb_data = []

for item in RAW_RESTAURANT_DATA:
    dishes = item.get("menu", {}).get("dishes", [])
    drinks = item.get("menu", {}).get("drinks", [])
    menu_items = dishes + drinks
    
    entry = {
        "id": item["id"],
        "name": item["name"],
        "city": item["city"],
        "cuisine": item["cuisine"],
        "text": item["text"],
        "menu": json.dumps(item["menu"]),
        "vector_menu": model.encode(" ".join(menu_items)).tolist(),
        "vector_dishes": model.encode(" ".join(dishes)).tolist(),
        "vector_drinks": model.encode(" ".join(drinks)).tolist(),
        "vector_city": model.encode(item["city"]).tolist(),
        "vector_cuisine": model.encode(item["cuisine"]).tolist(),
        "vector_restaurant": model.encode(item["name"]).tolist()
    }
    lancedb_data.append(entry)

TABLE_NAME = "restaurants"
db = lancedb.connect(DB_PATH)
try:
    table = db.create_table(
        TABLE_NAME,
        data=lancedb_data,
        mode="overwrite"
    )
    print(f"Databasen {TABLE_NAME} är skapad och uppdaterad.")
except Exception as e:
    print(f"Fel vid skapande: {e}. Försöker öppna tabellen.")
    table = db.open_table(TABLE_NAME)