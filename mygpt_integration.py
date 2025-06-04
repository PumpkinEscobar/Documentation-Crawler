from openai import OpenAI
import requests
import json
from typing import List, Dict, Any

class DocSearchTool:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

    def search_docs(self, query: str, top_k: int = 3) -> str:
        """Search the documentation and return formatted results"""
        try:
            response = requests.post(
                f"{self.api_url}/search",
                json={
                    "query": query,
                    "top_k": top_k
                }
            )
            response.raise_for_status()
            results = response.json()
            
            if not results["results"]:
                return "No relevant documentation found."
            
            # Format results into a nice text block
            formatted_results = []
            for r in results["results"]:
                formatted_results.append(
                    f"Source: {r['source']}\n"
                    f"Title: {r['title']}\n"
                    f"URL: {r['url']}\n"
                    f"Content: {r['content']}\n"
                )
            
            return "\n---\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching documentation: {str(e)}"

def create_chat_completion(
    client: OpenAI,
    messages: List[Dict[str, str]],
    doc_search: DocSearchTool
) -> str:
    """Create a chat completion with documentation support"""
    
    # Extract the user's question
    user_query = messages[-1]["content"]
    
    # Search documentation for relevant information
    doc_results = doc_search.search_docs(user_query)
    
    # Add documentation context to the messages
    messages_with_context = [
        {"role": "system", "content": """You are a helpful AI assistant with access to official AI documentation.
When answering questions about AI services, use the documentation provided to ensure accurate information.
Always cite your sources when using information from the documentation."""},
        {"role": "system", "content": f"Relevant documentation:\n{doc_results}"}
    ] + messages

    # Get completion from OpenAI
    response = client.chat.completions.create(
        model="gpt-4",  # or your preferred model
        messages=messages_with_context
    )
    
    return response.choices[0].message.content

def main():
    # Initialize OpenAI client
    client = OpenAI()  # Make sure OPENAI_API_KEY is set in your environment
    
    # Initialize documentation search tool
    doc_search = DocSearchTool()
    
    print("Chat with AI (type 'quit' to exit)")
    messages = []
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
            
        messages.append({"role": "user", "content": user_input})
        
        response = create_chat_completion(client, messages, doc_search)
        print("\nAssistant:", response)
        
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 