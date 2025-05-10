from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Azure AI Search setup
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

# Input query
query = "What are the healh guiglines for the construction site?"

# Step 1: Retrieve top 3 chunks from Azure AI Search
results = search_client.search(query, top=3)
context = "\n".join([doc.get("merged_content", "") for doc in results])

# Step 2: Azure OpenAI client setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Step 3: Generate GPT-4 answer
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ],
    temperature=0.3,
    max_tokens=500
)

# Step 4: Print output
print("\n🧠 Answer:\n" + response.choices[0].message.content)
