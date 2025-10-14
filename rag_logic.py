import os
from dotenv import load_dotenv
import lancedb
from sentence_transformers import SentenceTransformer
from google import genai
import numpy as np

# Implementera LLM
# Validering med Pydantic och BaseModel-klasser
# Ge resultat efter plats - google maps places, hårdkoda in ens egen location


# --- SETUP ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
DB_PATH = "restaurants_and_food_db"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

client = genai.Client(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
db = lancedb.connect(DB_PATH)
table = db.open_table("restaurants_db")


# --- USER QUERY LOGIC ---
def get_user_query(prompt: str) -> tuple[str | None, str | None]:
    """
    Extraherar restaurang och maträtt från prompten.
    Returnerar alltid tuple: (restaurant_name, dish_name)
    """
    prompt = prompt.strip()
    if not prompt:
        print("Du måste skriva något för att fortsätta.")
        return None, None

    prompt_lower = prompt.lower()
    prompt_words = prompt_lower.split()

    # --- Läs in hela tables och ta ut namnen på varje rad ---
    # --- Försöker matcha query och listan över namn ---
    df = table.to_pandas()
    mask = df["name"].apply(lambda x: x.lower() in prompt_lower)
    record_row = df[mask]

    # Om restaurang hittas
    if not record_row.empty:
        record = record_row.iloc[0].to_dict()
    else:
        # Fallback: vektorsök
        all_dishes = [dish.lower() for dishes in df["menu"].apply(lambda m: m.get("dishes", []) if m else []) for dish in dishes]
        
        mentioned_dishes = [d for d in all_dishes if any(word in d.lower() for word in prompt_words)]
        mentioned_dishes = list(dict.fromkeys(mentioned_dishes))
        
        if mentioned_dishes:
            search_query = " ".join(mentioned_dishes)
        else:
            print("Kan inte hitta restaurang med denna rätt")
            return None, None
        
        try:
            record = vectorize(search_query)
        except Exception as e:
            print(f"Kunde inte utföra vektorsök: {e}")
            return None, None
    
    restaurant_name = record["name"]

    # Hämta maträtter från menyn
    dishes = record.get("menu", {}).get("dishes", [])
    
    if isinstance(dishes, np.ndarray):
        dishes = dishes.tolist()

    # Kolla om någon maträtt nämns i prompten
    dish_name = None
    for dish in dishes:
        if dish.lower() in prompt_lower:
            dish_name = dish
            print(f"Du har beställt en {dish_name} från {restaurant_name}.")
            return record, dish_name
    
    # Om ingen maträtt nämns
    if not dish_name:
        if dishes:
            while(True):
            # Lista möjliga maträtter
                print(f"Maträtter som finns på {restaurant_name}: {', '.join(dishes)}")
                
                # Fråga användaren
                dish_name_input = input(f"Vad vill du äta på {restaurant_name}? Ange maträtt: ").strip()
                # 
                if not dish_name_input:
                    os.system("cls")
                    print(f"Du har valt {restaurant_name} utan att ange maträtt.")
                    continue
                # 
                if dish_name_input.lower() in [d.lower() for d in dishes]:
                    dish_name = dish_name_input
                    print(f"Du har beställt en {dish_name} från {restaurant_name}.")
                    return record, dish_name
                else:
                    print(f"Rätten '{dish_name_input}' finns tyvärr inte.")
                    continue
        else:
            print(f"Du har valt {restaurant_name}. Ingen meny finns tillgänglig.")
            dish_name = None


def vectorize(search_query: str):
    query_vector = embedding_model.encode(search_query).tolist()
    results = table.search(query_vector).limit(5).to_list()
    best = min(results, key=lambda r: r.get("_distance", 1.0))
    record = best
    return record