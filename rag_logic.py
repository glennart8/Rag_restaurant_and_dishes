import os
import json
from dotenv import load_dotenv
import lancedb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from models import PromptStructure
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Implementera LLM för at tutvinna väsentlig information, typ stad, maträtt, typ av kök
"""
1. En metod som tar in en query och skickar den till LLM
2. En instruktion där vi berättar till llm vad vi vill ha, i json
3. Uppdatera vektorer i databasen, en / attribut vi vill få ut
4. Först kollar vi mot databasen, om inget hittas - kolla mot vektordatabas
"""

# Validering med Pydantic och BaseModel-klasser
# Ge resultat efter plats - google maps places, hårdkoda in ens egen location


# --- SETUP ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
DB_PATH = "restaurants_and_food_db"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Initialiserar klienter och databas (se till att databasen/tabellen finns!)
client = genai.Client(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
db = lancedb.connect(DB_PATH)
table = db.open_table("restaurants")


def extract_entities_llm(prompt: str) -> dict:
    system_instruction = (
        """
        Din uppgift är att agera som dataextraktionsexpert.
        Du ska extrahera fälten 'maträtter', 'drycker', 'typ av kök', 'stad' och 'restaurang' från texten och svara som JSON.
        Om någon av fälten inte finns med i texten lämna den tom.
        
        #Format
        {
            name: "restaurang",
            city: "stad",
            cuisine: "typ av kök",
            dishes: "maträtter",
            drinks: "drycker"
        }
        """
    )
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=PromptStructure
        )
    )
    
    return response.text  
    

# --- USER QUERY LOGIC ---
def get_user_query(prompt: str) -> tuple[str | None, str | None]:
    """
    Extraherar restaurang och maträtt från prompten.
    Returnerar alltid tuple: (restaurant_name, dish_name)
    """
    prompt = prompt.strip()

    prompt_lower = prompt.lower()
    prompt_words = prompt_lower.split()

    # --- Kollar flrst om en restaurang nämns och finns i lsitan över restauranger---
    df = table.to_pandas()
    mask = df["name"].apply(lambda x: x.lower() in prompt_lower)
    record_row = df[mask]

    # Om restaurang hittas
    if not record_row.empty:
        record = record_row.iloc[0].to_dict()
    # Om inte, kolla maträtt i stället
    else:
        # Fallback: vektorsök
        # Går igenom alla menyer och skapar en lista över ALLA maträtter i ALLA restauranger
        all_dishes = [dish.lower() for dishes in df["menu"].apply(lambda m: m.get("dishes", []) if m else []) for dish in dishes]
        
        # kollar om någon av rätterna i all_dishes nämns i queryn
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
    
    # Slå ihop drycker och maträtter till en MENY från vald restaurang
    try:
        dishes = record.get("menu", {}).get("dishes", []).tolist()
        drinks = record.get("menu", {}).get("drinks", []).tolist()
        menu = dishes + drinks
        
    except Exception as e:
        print(f"Ingen meny funnen {e}")
        return None, None
    
    # Kolla om någon maträtt som nämns i prompten finns i bland restaurangens maträtt
    dish_name = None
    for dish in menu:
        if dish.lower() in prompt_lower:
            dish_name = dish
            print(f"Du har beställt en {dish_name} från {restaurant_name}.")
            return record, dish_name
    
    # Om ingen maträtt nämns
    if not dish_name:
            while(True):
            # Lista möjliga maträtter
                print(f"Menyn på:  {restaurant_name}: {', '.join(menu)}")
                
                # Fråga användaren
                dish_name_input = input(f"Vad vill du äta på {restaurant_name}? Ange maträtt: ").strip()
                # 
                if not dish_name_input:
                    os.system("cls")
                    print(f"Du har valt {restaurant_name} utan att ange maträtt.")
                    continue
                # 
                if dish_name_input.lower() in [d.lower() for d in menu]:
                    # Hittar en match
                    dish_name = dish_name_input
                    print(f"Du har beställt en {dish_name} från {restaurant_name}.")
                    return record, dish_name
                else:
                    # Ingen match
                    print(f"Rätten '{dish_name_input}' finns tyvärr inte.")
                    continue


def vectorize(search_query: str):
    query_vector = embedding_model.encode(search_query).tolist()
    results = table.search(query_vector).limit(5).to_list()
    best = min(results, key=lambda r: r.get("_distance", 1.0))
    record = best
    # 
    print([(r.get("name"), r.get("city"), r.get("_distance")) for r in results])
    return record

def multi_vector_search(table, vectors, weights, top_n=10):
    """
    table: LanceDB table
    vectors: list of tuples (col_name, query_vector)
    weights: list of floats (normalized)
    """
    all_scores = []

    for row in table:
        score = 0
        for (col_name, query_vec), weight in zip(vectors, weights):
            db_vec = np.array(row.get(col_name))
            sim = cosine_similarity([query_vec], [db_vec])[0][0]
            score += sim * weight
        all_scores.append((score, row))

    all_scores.sort(key=lambda x: x[0], reverse=True)
    return all_scores[:top_n]

def vectorize_fn(text_or_list):
    if isinstance(text_or_list, list):
        if not text_or_list:
            return [0.0] * 384  # fallback för MiniLM
        embeddings = [embedding_model.encode(t) for t in text_or_list]
        return np.mean(np.array(embeddings), axis=0).tolist()
    else:
        return embedding_model.encode(text_or_list).tolist()
    
    
    
if __name__ == "__main__":
    #prompt = "Jag vill äta kinesiskt, gärna mapo tofu i Göteborg"
    prompt = "Jag vill äta ryggbiff och dricka en öl i göteborg,"

    data = extract_entities_llm(prompt)
    data = json.loads(data)
    # Extrahera sökbegrepp (enkelt exempel)
    restaurant_name = data.get("name")
    city = data.get("city")
    cuisine = data.get("cuisine")
    dishes = data.get("dishes") or []
    drinks = data.get("drinks") or []
    menu = (dishes or []) + (drinks or [])

    # Skapa query-vektorer
    vectors = []
    weights = []

    print(f"Söker med {data.get("city")} {data.get("dishes")}")
    print(f"{data.get("cuisine")} {data.get("name")} {data.get("drinks")}")
    print(f"{menu=}")    # Borde ta bort restaurang och city och separera meny till dish och drinks
    # if drinks and dishes:
    #     vectors.append(("vector_menu", vectorize_fn(menu)))
    #     weights.append(0.7)
    if city:
        vectors.append(("vector_city", vectorize_fn(city)))
        weights.append(0.3)
    if cuisine:
        vectors.append(("vector_cuisine", vectorize_fn(cuisine)))
        weights.append(0.2)
    if restaurant_name:
        vectors.append(("vector_restaurant", vectorize_fn(restaurant_name)))
        weights.append(0.3)
    if menu:
        if drinks:
            vectors.append(("vector_drinks", vectorize_fn(drinks)))
            weights.append(0.2)
        if dishes:
            vectors.append(("vector_dishes", vectorize_fn(dishes)))
            weights.append(0.7)
    

    # Normalisera vikter
    total = sum(weights)
    weights = [w / total for w in weights]

    test = table.to_pandas().to_dict(orient="records")
    # Sök
    results = multi_vector_search(test, vectors, weights)

    # Visa resultat
    for r in results[:5]:
        print(f"score={r[0]:.4f} name = {r[1].get("name")}")
