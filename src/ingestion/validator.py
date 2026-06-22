import re
import hashlib
import logging

logger = logging.getLogger(__name__)

# Try importing langdetect
HAS_LANGDETECT = False
try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    pass

class DataValidator:
    def __init__(self, allowed_languages=None):
        self.allowed_languages = allowed_languages or ['en']
        # Set of common english words as robust fallback language check
        self.english_keywords = {"the", "and", "you", "that", "was", "for", "are", "with", "his", "they", "this", "have", "but", "not"}

    def is_english_fallback(self, text: str) -> bool:
        """Fallback check for English using stop word occurrences."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return False
        eng_count = sum(1 for w in words if w in self.english_keywords)
        # If at least 5% or 1 word of the words are common english stopwords, we classify as english
        return eng_count >= 1 or (eng_count / len(words)) >= 0.05

    def validate_record(self, record: dict) -> dict:
        """
        Validates a single record dictionary.
        Returns a dict indicating if it is valid and any validation issues found.
        """
        issues = []
        is_corrupted = False
        is_duplicate = False
        unsupported_lang = False
        
        # 1. Missing Value / Null check
        text = record.get("text", "")
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            issues.append("Missing or empty text content.")
            is_corrupted = True
            
        # 2. Corrupted record check (extreme lengths or characters)
        if not is_corrupted:
            clean_text = text.strip()
            # If text has zero alphanumeric characters or is just punctuation
            if not re.search(r'[a-zA-Z0-9]', clean_text):
                issues.append("Corrupted content (contains no alphanumeric characters).")
                is_corrupted = True
            elif len(clean_text) < 5:
                issues.append("Record too short.")
                is_corrupted = True

        # 3. Language check
        if not is_corrupted:
            lang = "unknown"
            if HAS_LANGDETECT:
                try:
                    lang = detect(clean_text)
                except Exception:
                    pass
            
            # Run fallback if langdetect failed or is missing
            if lang == "unknown" or not HAS_LANGDETECT:
                is_eng = self.is_english_fallback(clean_text)
                lang = 'en' if is_eng else 'other'
                
            if lang not in self.allowed_languages:
                issues.append(f"Unsupported language detected: {lang}")
                unsupported_lang = True
                
        return {
            "is_valid": not (is_corrupted or unsupported_lang),
            "is_corrupted": is_corrupted,
            "unsupported_lang": unsupported_lang,
            "issues": issues
        }

    def validate_batch(self, records: list, existing_hashes: set = None) -> dict:
        """
        Validates a batch of records. Checks for duplicates.
        Returns clean records, filtered records, and a validation summary report.
        """
        existing_hashes = existing_hashes or set()
        validated_records = []
        corrupted_count = 0
        duplicate_count = 0
        unsupported_lang_count = 0
        valid_count = 0
        
        report_details = []
        
        for idx, r in enumerate(records):
            val_res = self.validate_record(r)
            
            if not val_res["is_valid"]:
                if val_res["is_corrupted"]:
                    corrupted_count += 1
                if val_res["unsupported_lang"]:
                    unsupported_lang_count += 1
                
                report_details.append({
                    "index": idx,
                    "issues": val_res["issues"],
                    "text_preview": r.get("text", "")[:50] + "..."
                })
                continue
                
            # Duplicate detection using md5 hash of lowercase text
            text_bytes = r["text"].strip().lower().encode('utf-8')
            text_hash = hashlib.md5(text_bytes).hexdigest()
            
            if text_hash in existing_hashes:
                duplicate_count += 1
                report_details.append({
                    "index": idx,
                    "issues": ["Duplicate record detected."],
                    "text_preview": r["text"][:50] + "..."
                })
                continue
                
            existing_hashes.add(text_hash)
            valid_count += 1
            validated_records.append(r)
            
        total = len(records)
        quality_score = (valid_count / total) * 100 if total > 0 else 100.0
        
        report = {
            "total_records": total,
            "valid_records": valid_count,
            "corrupted_records": corrupted_count,
            "duplicate_records": duplicate_count,
            "unsupported_language_records": unsupported_lang_count,
            "data_quality_score": round(quality_score, 2),
            "issues_log": report_details
        }
        
        return {
            "valid_records": validated_records,
            "report": report
        }
