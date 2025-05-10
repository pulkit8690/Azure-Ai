import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load .env file
load_dotenv()

# Get connection string from .env
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Define your container and file to upload
container_name = "pdf-container"            # Your container in Azure
file_path = "./pdfs/Health_Safety_Guidelines.pdf"                   # Local file path
blob_name = os.path.basename(file_path)     # Blob name in Azure

# Create the BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Create a client for your specific blob
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# Upload the file
with open(file_path, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)

print(f"✅ Uploaded '{file_path}' to Azure Blob container '{container_name}'.")
