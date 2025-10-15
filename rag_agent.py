from rag_logic import get_user_query

DB_PATH = "restaurants_and_food_db"

def run_rag_agent():
    """
    1. Frågar vad användaren vill äta
    2. Hämtar namn på restaurang och maträtt
    3. Skapar context innehållande vald restaurang
    4. Anropar run_gemini_query som returnerar den validerade outputen
    5. Printar den validerade outputen
    
    """
    print("=== RAG-AGENT STARTAD ===")

    while True:
        prompt = input("\nVad vill du äta? (q för att avsluta): ").strip()
        if prompt.lower() == "q":
            break

        if not prompt:
            print("Du måste skriva något för att fortsätta.")
            continue
        
        # Hämta restaurang och maträtt
        record, dish = get_user_query(prompt)
        if not record:
            continue

        print("\n--- RESULTAT ---")
        print(f"Restaurang: {record["name"]}")
        print(f"Stad: {record["city"]}")
        print(f"Köpt maträtt: {dish}")

if __name__ == "__main__":
    run_rag_agent()
