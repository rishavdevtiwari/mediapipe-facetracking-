import os
from google import genai
from dotenv import load_dotenv

def main():
    # Load API Key from .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Please verify your .env file.")

    client = genai.Client(api_key=api_key)
    
    print("Initializing conversational session...")
    chat = client.chats.create(model='gemini-2.5-flash')
    
    print("\n🤖 Chatbot Online! Type 'exit' or 'quit' to end the session.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            # Send message and maintain context history
            response = chat.send_message(user_input)
            print(f"\nAI: {response.text}")
            
        except Exception as e:
            print(f"\n⚠️ Error encountered: {e}")

if __name__ == "__main__":
    main()