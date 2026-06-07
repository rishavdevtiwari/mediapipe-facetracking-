import os
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool
from dotenv import load_dotenv

def main():
    # setup-
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Please verify your .env file.")

    client = genai.Client(api_key=api_key)
    
    #Google Search Tool 
    search_config = GenerateContentConfig(
        system_instruction="You are a live web research agent. When the user gives you a topic or word, search the web for the most recent, relevant information and provide a concise summary of what you found.",
        tools=[Tool(google_search=GoogleSearch())]
    )
    
    print("Connecting to Gemini and initializing web tools...")
    chat = client.chats.create(
        model='gemini-2.5-flash', 
        config=search_config
    )
    
    print("\n🌐 Web Search Agent Online!")
    print("Type any word, topic, or question to fetch live web contents. Type 'exit' to quit.")
    print("-" * 60)
    
    # -convo loop
    while True:
        try:
            user_input = input("\nSearch Target: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down search agent. Goodbye!")
                break
                
            print("⏳ Scouring the web...")
            
            # decide if web search required
            response = chat.send_message(user_input)
            
            # summary of the web contents
            print(f"\nResults:\n{response.text}")
            
            print(f"\n🔍 Search Metadata: {response.candidates[0].grounding_metadata}")
            
            #  see the sources
            # by inspecting response.candidates[0].grounding_metadata
            
        except Exception as e:
            print(f"\n⚠️ Error encountered during search: {e}")

if __name__ == "__main__":
    main()