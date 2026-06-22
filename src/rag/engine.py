import os
import requests
import logging

logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self):
        # Memory storage: list of {"role": "user"/"assistant", "content": str}
        self.memory = {}

    def get_memory(self, session_id: str) -> list:
        if session_id not in self.memory:
            self.memory[session_id] = []
        return self.memory[session_id]

    def clear_memory(self, session_id: str):
        if session_id in self.memory:
            self.memory[session_id] = []

    def answer_question(self, 
                        question: str, 
                        retrieved_docs: list, 
                        openai_api_key: str = None, 
                        use_ollama: bool = False, 
                        ollama_model: str = "llama3.2",
                        session_id: str = "default",
                        ollama_host: str = None) -> dict:
        """
        Synthesizes an answer using the retrieved reviews and target LLM model.
        Supports citation tracking and conversation memory.
        """
        # If no reviews retrieved
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant reviews in the database to answer your question. Please make sure reviews are ingested and try again.",
                "sources": [],
                "engine_used": "No Engine (No Context)"
            }

        # 1. Format context with source metadata
        context_items = []
        for idx, doc in enumerate(retrieved_docs):
            meta = doc.get("metadata", {})
            user = meta.get("user", "Anonymous")
            source = meta.get("source", "Web")
            sentiment = meta.get("global_sentiment", "Neutral")
            category = meta.get("category", "General")
            date = meta.get("timestamp", "")
            
            context_items.append(
                f"[Source #{idx + 1}] | User: {user} | Platform: {source} | Date: {date} | Sentiment: {sentiment} | Category: {category}\n"
                f"Feedback Text: \"{doc['text']}\"\n"
            )
        
        context_str = "\n---\n".join(context_items)

        # 2. Get history from memory
        history = self.get_memory(session_id)
        history_str = ""
        if history:
            history_str = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history[-4:]]) # Last 2 rounds

        # 3. Formulate Prompt Template
        system_prompt = (
            "You are a Customer Voice Intelligence AI assistant. Your task is to analyze customer feedback and answer the user's questions.\n"
            "Guidelines:\n"
            "1. Answer the question FACTUALLY based ONLY on the provided Context of Customer Reviews below.\n"
            "2. Cite your sources by appending [Source #X] at the end of statements where applicable.\n"
            "3. If the context does not contain relevant information, say 'The retrieved reviews do not contain enough details to answer this question. However, based on the general feedback context...' and summarize the context.\n"
            "4. Structure your response with clear headings, bullet points, and positive/negative trends if applicable.\n"
        )
        
        prompt = (
            f"{system_prompt}\n"
            f"--- CONVERSATION HISTORY ---\n{history_str}\n"
            f"--- RETRIEVED CUSTOMER REVIEWS CONTEXT ---\n{context_str}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        # 4. Route generation to appropriate engine
        answer = ""
        engine_used = "Heuristic Fallback Engine"
        
        # Override with environment variables if present
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if openai_key and openai_key.strip() and openai_key != "sk-...":
            try:
                answer = self._query_openai(prompt, openai_key.strip())
                engine_used = "OpenAI GPT-4o-mini"
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                answer = f"**OpenAI API error ({e}).**\n\n*Falling back to local heuristic summary:*\n\n" + self._heuristic_generation(retrieved_docs, question)
        elif use_ollama:
            try:
                answer = self._query_ollama(prompt, ollama_model, ollama_host)
                engine_used = f"Ollama ({ollama_model})"
            except Exception as e:
                logger.error(f"Ollama server call failed: {e}")
                answer = f"**Ollama server connection failed ({e}). Please verify Ollama is running and accessible at host endpoint.**\n\n*Falling back to local heuristic summary:*\n\n" + self._heuristic_generation(retrieved_docs, question)
        else:
            answer = self._heuristic_generation(retrieved_docs, question)
            engine_used = "Local Heuristic Summarizer (No LLM)"

        # 5. Save turn to memory
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        
        # Return structured output with citations list
        sources_metadata = []
        for idx, doc in enumerate(retrieved_docs):
            meta = doc.get("metadata", {})
            sources_metadata.append({
                "source_id": idx + 1,
                "text": doc["text"],
                "user": meta.get("user", "Anonymous"),
                "source": meta.get("source", "Web"),
                "sentiment": meta.get("global_sentiment", "Neutral"),
                "timestamp": meta.get("timestamp", "")
            })

        return {
            "answer": answer,
            "sources": sources_metadata,
            "engine_used": engine_used
        }

    def _query_openai(self, prompt: str, api_key: str) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=20
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _query_ollama(self, prompt: str, model: str, host: str = None) -> str:
        base_host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        base_host = base_host.rstrip('/')
        url = f"{base_host}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["response"]

    def _heuristic_generation(self, reviews: list, question: str) -> str:
        """
        Synthesizes a rule-based bulleted analysis from reviews to act as a fallback.
        """
        positives = [r for r in reviews if r.get("metadata", {}).get("global_sentiment") == "POSITIVE"]
        negatives = [r for r in reviews if r.get("metadata", {}).get("global_sentiment") == "NEGATIVE"]
        
        output = f"Here is a summary analysis based on the **{len(reviews)} retrieved client reviews** matching your query:\n\n"
        
        if negatives:
            output += "### ⚠️ Critical Pain Points & Concerns\n"
            for i, r in enumerate(negatives[:3]):
                meta = r.get("metadata", {})
                user = meta.get("user", "User")
                source = meta.get("source", "Platform")
                output += f"- \"*{r['text']}*\" — **{user}** on {source} [Source #{reviews.index(r) + 1}]\n"
            output += "\n"
            
        if positives:
            output += "### ✅ Positive Praise & Features Highlighted\n"
            for i, r in enumerate(positives[:3]):
                meta = r.get("metadata", {})
                user = meta.get("user", "User")
                source = meta.get("source", "Platform")
                output += f"- \"*{r['text']}*\" — **{user}** on {source} [Source #{reviews.index(r) + 1}]\n"
            output += "\n"
            
        output += "### 💡 Recommendation\n"
        output += "Reviewing these records shows that product reliability and response times are the primary drivers of customer sentiment.\n\n"
        output += "*Note: Configure your OpenAI key or run Ollama locally to synthesize generative AI reviews.*"
        
        return output
