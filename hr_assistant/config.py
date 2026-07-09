import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    # ai
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    # modello ai per embeddings
    MODEL_NAME = "text-embedding-3-small"
    # cartelle database
    PERSISTENT_DIR = "data/chromadb"
    # collezzione
    COLLECTION_NAME = "CVs"
    # cartelle documenti
    DOCUMENTS_DIR = "resumes"

    # ollama
    # AI_API_URL = "http://localhost:11434/v1"
    # AI_API_KEY = "ollama"
    # LLM_MODEL = "llama3.2"
    # LLM_MODEL_LOW = "llama3.2"
    
    AI_API_URL = "https://api.openai.com/v1/"
    AI_API_KEY = os.getenv("AI_API_KEY")
    LLM_MODEL = "gpt-4o"
    LLM_MODEL_LOW = "gpt-4o-mini"
   
