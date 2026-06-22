import subprocess
import sys
import os

def check_dependencies():
    """
    Checks and downloads necessary NLTK and spaCy models before startup.
    """
    print("Checking system NLP models and packages...")
    
    # 1. Check spaCy core model
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("-> spaCy 'en_core_web_sm' is installed.")
    except (ImportError, OSError):
        print("-> spaCy model 'en_core_web_sm' not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            print("-> Successfully downloaded spaCy model.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading spaCy model: {e}")
            
    # 2. Check NLTK stopwords/tokenizer
    try:
        import nltk
        for package in ['stopwords', 'punkt']:
            try:
                nltk.data.find(f'corpora/{package}' if package == 'stopwords' else f'tokenizers/{package}')
            except LookupError:
                print(f"-> NLTK package '{package}' not found. Downloading...")
                nltk.download(package, quiet=True)
        print("-> NLTK packages are verified.")
    except ImportError:
        print("NLTK package is missing. Please run 'pip install -r requirements.txt'")

def check_db_initialization():
    """
    Initializes SQL databases locally.
    """
    print("Initializing Database tables...")
    try:
        # Import to trigger engines setup and table schema creation
        from src.database.connection import init_db
        init_db()
        print("-> Database initialized successfully.")
    except Exception as e:
        print(f"Warning: Database initialization encountered warnings ({e}). Details will be logged in runtime.")

def launch_dashboard():
    """
    Launches the Streamlit application.
    """
    os.makedirs("data", exist_ok=True)
    app_path = os.path.join("src", "dashboard", "app.py")
    print(f"Launching Streamlit interface: {app_path}")
    try:
        subprocess.run(["streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    except FileNotFoundError:
        print("Error: 'streamlit' command not found. Verify your virtual environment is active and packages are installed.")

if __name__ == "__main__":
    check_dependencies()
    check_db_initialization()
    print("--- Setup Complete ---")
    launch_dashboard()
