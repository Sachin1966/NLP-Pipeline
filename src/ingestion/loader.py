import os
import json
import logging
import pandas as pd
from pypdf import PdfReader
from docx import Document
from PIL import Image

logger = logging.getLogger(__name__)

# Check for OCR capabilities
HAS_OCR = False
OCR_ENGINE = None

try:
    import pytesseract
    # Test if tesseract is installed
    pytesseract.get_tesseract_version()
    HAS_OCR = True
    OCR_ENGINE = "tesseract"
except Exception:
    try:
        import easyocr
        # Init reader in english
        _ = easyocr.Reader(['en'], gpu=False)
        HAS_OCR = True
        OCR_ENGINE = "easyocr"
    except Exception:
        logger.warning("Neither pytesseract nor easyocr is available. OCR requests will fall back to mockup metadata read.")

class DocumentIngester:
    def __init__(self):
        pass

    def load_file(self, filepath: str) -> list:
        """
        Loads files and returns a list of dictionaries with at least a 'text' key.
        Other keys can include metadata like 'source', 'user', 'timestamp'.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            return self._load_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            return self._load_excel(filepath)
        elif ext == '.json':
            return self._load_json(filepath)
        elif ext == '.txt':
            return self._load_txt(filepath)
        elif ext == '.pdf':
            return self._load_pdf(filepath)
        elif ext == '.docx':
            return self._load_docx(filepath)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            return self._load_image_ocr(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_csv(self, filepath: str) -> list:
        df = pd.read_csv(filepath)
        return self._df_to_records(df, filepath)

    def _load_excel(self, filepath: str) -> list:
        df = pd.read_excel(filepath)
        return self._df_to_records(df, filepath)

    def _load_json(self, filepath: str) -> list:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            # Ensure text field is present
            records = []
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    records.append({
                        "text": item.get("text", str(item)),
                        "source": item.get("source", "JSON Upload"),
                        "user": item.get("user", f"user_{idx}"),
                        "timestamp": item.get("timestamp", pd.Timestamp.now().isoformat())
                    })
                else:
                    records.append({
                        "text": str(item),
                        "source": "JSON Upload",
                        "user": f"user_{idx}",
                        "timestamp": pd.Timestamp.now().isoformat()
                    })
            return records
        elif isinstance(data, dict):
            # If single dict, check for list under common keys
            for k, v in data.items():
                if isinstance(v, list):
                    return self._load_json_data(v, filepath)
            return [{
                "text": data.get("text", str(data)),
                "source": data.get("source", "JSON Upload"),
                "user": data.get("user", "user_0"),
                "timestamp": data.get("timestamp", pd.Timestamp.now().isoformat())
            }]
        return []

    def _load_json_data(self, data_list: list, filepath: str) -> list:
        records = []
        for idx, item in enumerate(data_list):
            if isinstance(item, dict):
                records.append({
                    "text": item.get("text", str(item)),
                    "source": item.get("source", os.path.basename(filepath)),
                    "user": item.get("user", f"user_{idx}"),
                    "timestamp": item.get("timestamp", pd.Timestamp.now().isoformat())
                })
        return records

    def _load_txt(self, filepath: str) -> list:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Split by empty lines or just keep as one record
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        records = []
        for idx, p in enumerate(paragraphs):
            records.append({
                "text": p,
                "source": os.path.basename(filepath),
                "user": f"txt_client_{idx}",
                "timestamp": pd.Timestamp.now().isoformat()
            })
        return records

    def _load_pdf(self, filepath: str) -> list:
        reader = PdfReader(filepath)
        text_content = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        
        full_text = "\n".join(text_content)
        # Parse into logical reviews/paragraphs
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [full_text]
            
        return [{
            "text": p,
            "source": os.path.basename(filepath),
            "user": "pdf_extractor",
            "timestamp": pd.Timestamp.now().isoformat()
        } for p in paragraphs if len(p) > 10]

    def _load_docx(self, filepath: str) -> list:
        doc = Document(filepath)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return [{
            "text": p,
            "source": os.path.basename(filepath),
            "user": "docx_extractor",
            "timestamp": pd.Timestamp.now().isoformat()
        } for p in paragraphs if len(p) > 10]

    def _load_image_ocr(self, filepath: str) -> list:
        """Extracts text from images using available OCR engine."""
        text = ""
        if HAS_OCR:
            try:
                if OCR_ENGINE == "tesseract":
                    text = pytesseract.image_to_string(Image.open(filepath))
                elif OCR_ENGINE == "easyocr":
                    import easyocr
                    reader = easyocr.Reader(['en'], gpu=False)
                    results = reader.readtext(filepath)
                    text = " ".join([res[1] for res in results])
            except Exception as e:
                logger.error(f"OCR reading failed: {e}")
                text = f"[OCR ERROR] Could not read file contents: {e}"
        else:
            text = f"[OCR NOT AVAILABLE] Uploaded image {os.path.basename(filepath)} could not be read because OCR dependencies are missing."
            
        return [{
            "text": text,
            "source": f"OCR: {os.path.basename(filepath)}",
            "user": "ocr_reader",
            "timestamp": pd.Timestamp.now().isoformat()
        }]

    def _df_to_records(self, df: pd.DataFrame, filepath: str) -> list:
        records = []
        # Find column that contains text content
        text_cols = [c for c in df.columns if c.lower() in ['text', 'review', 'comment', 'body', 'feedback', 'message', 'ticket']]
        text_col = text_cols[0] if text_cols else None
        
        if not text_col:
            # Fallback to first string column or first column
            string_cols = df.select_dtypes(include=['object']).columns
            text_col = string_cols[0] if len(string_cols) > 0 else df.columns[0]
            
        # Select optional metadata columns
        user_col = [c for c in df.columns if c.lower() in ['user', 'username', 'customer', 'name', 'author']][0] if [c for c in df.columns if c.lower() in ['user', 'username', 'customer', 'name', 'author']] else None
        source_col = [c for c in df.columns if c.lower() in ['source', 'channel', 'platform']][0] if [c for c in df.columns if c.lower() in ['source', 'channel', 'platform']] else None
        time_col = [c for c in df.columns if c.lower() in ['timestamp', 'date', 'created_at', 'time']][0] if [c for c in df.columns if c.lower() in ['timestamp', 'date', 'created_at', 'time']] else None
        rating_col = [c for c in df.columns if c.lower() in ['rating', 'score', 'stars']][0] if [c for c in df.columns if c.lower() in ['rating', 'score', 'stars']] else None
        category_col = [c for c in df.columns if c.lower() in ['category', 'type', 'label']][0] if [c for c in df.columns if c.lower() in ['category', 'type', 'label']] else None
        
        for idx, row in df.iterrows():
            text_val = str(row[text_col]) if not pd.isna(row[text_col]) else ""
            if not text_val.strip():
                continue
                
            records.append({
                "text": text_val,
                "source": str(row[source_col]) if source_col and not pd.isna(row[source_col]) else os.path.basename(filepath),
                "user": str(row[user_col]) if user_col and not pd.isna(row[user_col]) else f"user_{idx}",
                "timestamp": str(row[time_col]) if time_col and not pd.isna(row[time_col]) else pd.Timestamp.now().isoformat(),
                "rating": float(row[rating_col]) if rating_col and not pd.isna(row[rating_col]) else None,
                "category": str(row[category_col]) if category_col and not pd.isna(row[category_col]) else None
            })
        return records
