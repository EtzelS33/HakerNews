import os
import sys
import platform
import requests
import feedparser
import json
import time
import argparse
import webbrowser
import glob
import hashlib
import base64
import io
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from gtts import gTTS
from PIL import Image
from tqdm import tqdm
from playwright.sync_api import sync_playwright

# --- CONFIGURAZIONI GENERALI ---
SETTINGS_FILE = "llm_settings.json"
OUTPUT_DIR = "hn_analysis"
AUDIO_DIR = "hn_audio"
SCREENSHOT_DIR = "hn_screenshots"
TEMPLATE_FILE = "hn_template.html"

# --- MACRO TAGS (unificato) ---
MACRO_TAGS = {
    "🤖 AI & Machine Learning": [
        "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
        "neural network", "llm", "large language model", "gpt", "generative ai",
        "computer vision", "nlp", "natural language processing", "transformer",
        "diffusion model", "stable diffusion", "midjourney", "dall-e", "chatgpt",
        "claude", "gemini", "copilot", "ai ethics", "agi", "reinforcement learning",
        "tensorflow", "pytorch", "scikit-learn", "hugging face"
    ],
    "💻 Programmazione & DevOps": [
        "programming", "coding", "software development", "javascript", "python",
        "java", "c++", "rust", "go", "typescript", "react", "vue", "angular",
        "docker", "kubernetes", "devops", "ci/cd", "git", "github", "gitlab",
        "api", "microservices", "serverless", "cloud native", "agile", "scrum"
    ],
    "🔒 Sicurezza & Privacy": [
        "security", "cybersecurity", "privacy", "encryption", "cryptography",
        "vulnerability", "exploit", "hacking", "penetration testing", "firewall",
        "vpn", "tor", "gdpr", "data protection", "password", "authentication",
        "2fa", "zero trust", "ransomware", "malware", "phishing"
    ],
    "🌐 Web & Cloud": [
        "web development", "frontend", "backend", "full stack", "aws", "azure",
        "google cloud", "gcp", "cloud computing", "saas", "paas", "iaas",
        "cdn", "hosting", "domain", "ssl", "https", "rest api", "graphql",
        "websocket", "pwa", "jamstack", "edge computing"
    ],
    "📊 Data & Database": [
        "database", "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        "elasticsearch", "data science", "big data", "data analytics", "etl",
        "data warehouse", "data lake", "hadoop", "spark", "kafka", "data visualization",
        "tableau", "power bi", "pandas", "numpy"
    ],
    "🚀 Startup & Business": [
        "startup", "entrepreneurship", "venture capital", "vc", "funding",
        "ipo", "acquisition", "merger", "business model", "saas business",
        "b2b", "b2c", "growth hacking", "product market fit", "unicorn",
        "y combinator", "accelerator", "pitch deck", "mvp", "lean startup"
    ],
    "🔬 Scienza & Ricerca": [
        "science", "research", "physics", "chemistry", "biology", "medicine",
        "quantum computing", "quantum physics", "astronomy", "space", "nasa",
        "climate change", "renewable energy", "biotechnology", "genetics",
        "crispr", "vaccine", "drug discovery", "clinical trial", "peer review"
    ],
    "🎮 Gaming & VR/AR": [
        "gaming", "video games", "game development", "unity", "unreal engine",
        "vr", "virtual reality", "ar", "augmented reality", "mr", "mixed reality",
        "metaverse", "oculus", "quest", "steamvr", "game design", "indie game",
        "aaa game", "mobile gaming", "esports"
    ],
    "⚡ Hardware & IoT": [
        "hardware", "cpu", "gpu", "semiconductor", "chip", "processor",
        "raspberry pi", "arduino", "iot", "internet of things", "embedded",
        "fpga", "asic", "sensor", "robotics", "3d printing", "maker",
        "diy electronics", "pcb", "firmware"
    ],
    "💰 Crypto & Blockchain": [
        "cryptocurrency", "bitcoin", "ethereum", "blockchain", "defi",
        "nft", "web3", "smart contract", "dao", "mining", "wallet",
        "exchange", "stablecoin", "altcoin", "consensus", "proof of work",
        "proof of stake", "layer 2", "polygon", "solana"
    ]
}


# --- FUNZIONI OPERATIVE (da hn_analyzer_win, adattate cross-platform) ---
def sanitize_filename(name):
    name = name.split('?')[0]
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def get_macro_tags(tags):
    macro_tags_found = set()
    if not isinstance(tags, list):
        return []
    for tag in tags:
        tag_lower = tag.lower()
        for macro_tag, keywords in MACRO_TAGS.items():
            for keyword in keywords:
                if tag_lower == keyword.lower() or keyword.lower() in tag_lower or tag_lower in keyword.lower():
                    macro_tags_found.add(macro_tag)
                    break
    return list(macro_tags_found)

def generate_audio(text, filename, title=""):
    try:
        full_text = f"{title}. {text}" if title else text
        tts = gTTS(text=full_text, lang='it', slow=False)
        audio_path = os.path.join(AUDIO_DIR, filename)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"Errore nella generazione audio per {filename}: {e}")
        return None

def capture_screenshot(url, filename):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(2000)
            screenshot_path = os.path.join(SCREENSHOT_DIR, filename)
            page.screenshot(path=screenshot_path, full_page=False)
            browser.close()
            thumbnail_base64 = create_thumbnail_base64(screenshot_path)
            return screenshot_path, thumbnail_base64
    except Exception as e:
        print(f"Errore nella cattura screenshot per {url}: {e}")
        return None, None

def create_thumbnail_base64(image_path, max_width=300):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = height / width
            new_width = min(width, max_width)
            new_height = int(new_width * aspect_ratio)
            thumbnail = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            if thumbnail.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', thumbnail.size, (255, 255, 255))
                background.paste(thumbnail, mask=thumbnail.split()[-1] if thumbnail.mode == 'RGBA' else None)
                thumbnail = background
            thumbnail.save(buffer, format='JPEG', quality=85, optimize=True)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"Errore nella creazione della miniatura: {e}")
        return None

def setup_environment():
    for directory in [OUTPUT_DIR, AUDIO_DIR, SCREENSHOT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print(f"Ambiente configurato. Output in: {OUTPUT_DIR}, {AUDIO_DIR}, {SCREENSHOT_DIR}")

def fetch_hackernews_feed(url="https://news.ycombinator.com/rss"):
    try:
        feed = feedparser.parse(url)
        print(f"Recuperati {len(feed.entries)} articoli dal feed di Hacker News")
        return feed.entries
    except Exception as e:
        print(f"Errore nel recupero del feed: {e}")
        return []

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:8000]
    except Exception as e:
        return f"Errore nell'estrazione del testo: {str(e)}"

def get_llm_config():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return None

def set_llm_config(config):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def fetch_available_models(provider, api_url):
    try:
        if provider == "ollama":
            check_url = urlparse(api_url)._replace(path="/api/tags").geturl()
            response = requests.get(check_url, timeout=10)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        elif provider == "lmstudio":
            check_url = urlparse(api_url)._replace(path="/v1/models").geturl()
            response = requests.get(check_url, timeout=10)
            response.raise_for_status()
            models = response.json().get('data', [])
            return [model['id'] for model in models]
    except requests.exceptions.RequestException as e:
        print(f"\nAvviso: Impossibile recuperare i modelli da {provider.upper()}. {e}")
    except Exception as e:
        print(f"\nAvviso: Errore imprevisto durante il recupero dei modelli: {e}")
    return []

def interactive_config(current_config=None):
    print("--- Configurazione LLM Interattiva ---")
    default_provider = current_config.get('llm_provider', 'ollama') if current_config else 'ollama'
    provider = input(f"Scegli il provider [ollama/lmstudio] (default: {default_provider}): ").strip().lower() or default_provider
    while provider not in ['ollama', 'lmstudio']:
        print("Provider non valido.")
        provider = input("Scegli tra [ollama/lmstudio]: ").strip().lower()
    default_urls = {
        'ollama': 'http://127.0.0.1:11434/api/generate',
        'lmstudio': 'http://127.0.0.1:1234/v1/chat/completions'
    }
    default_api_url = current_config.get('api_url', default_urls[provider]) if current_config else default_urls[provider]
    api_url = input(f"Inserisci l'URL dell'API (default: {default_api_url}): ").strip() or default_api_url
    print("\nRecupero modelli disponibili...")
    available_models = fetch_available_models(provider, api_url)
    model_name = None
    if available_models:
        print("Modelli disponibili rilevati:")
        for i, model in enumerate(available_models):
            print(f"  {i + 1}: {model}")
        last_model = current_config.get('model_name') if current_config else None
        default_choice = ""
        if last_model and last_model in available_models:
            default_choice = str(available_models.index(last_model) + 1)
        while not model_name:
            choice = input(f"Scegli un modello (numero) [default: {default_choice}]: ").strip() or default_choice
            try:
                model_index = int(choice) - 1
                if 0 <= model_index < len(available_models):
                    model_name = available_models[model_index]
                else:
                    print("Scelta non valida.")
            except ValueError:
                print("Inserisci un numero.")
    else:
        print("Nessun modello rilevato automaticamente. Procedere con inserimento manuale.")
        default_model = current_config.get('model_name', 'gemma3:latest') if current_config else 'gemma3:latest'
        model_name = input(f"Inserisci il nome del modello (default: {default_model}): ").strip() or default_model
    config = {
        'llm_provider': provider,
        'api_url': api_url,
        'model_name': model_name
    }
    set_llm_config(config)
    print("\nConfigurazione salvata.")
    return config

def process_with_llm(text, url, api_url, provider, model_name):
    prompt = f"""
    Analizza il seguente testo proveniente dall'URL: {url}
    TESTO:
    {text}

    Restituisci l'output ESCLUSIVAMENTE in formato JSON, senza alcuna formattazione markdown, commento o testo aggiuntivo.
    L'output deve essere un singolo oggetto JSON valido.

    La struttura JSON deve essere la seguente:
    {{
        "title": "Titolo dell'articolo, tradotto in italiano",
        "summary": "Un riassunto conciso in ITALIANO (massimo 100 parole) del contenuto. Il riassunto deve essere una singola stringa di testo senza caratteri di nuova riga.",
        "tags": ["elenco", "di", "3-7", "tag", "rilevanti"],
        "main_link": "URL principale menzionata nell'articolo o stringa vuota se non presente"
    }}

    ATTENZIONE: La tua risposta deve iniziare con '{{' e finire con '}}' e contenere solo il JSON.
    """
    headers = {"Content-Type": "application/json"}
    payload = {}
    if provider == "ollama":
        payload = {
            "model": model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
    elif provider == "lmstudio":
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": False
        }
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        if provider == "ollama":
            raw_response = result.get('response', '')
        elif provider == "lmstudio":
            raw_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not raw_response.strip():
            print(f"\n[AVVISO] Risposta vuota ricevuta da '{provider}'. Salto articolo.")
            return None
        single_line_response = raw_response.replace('\n', ' ').replace('\r', '')
        json_start = single_line_response.find('{')
        if json_start == -1:
            print(f"\n[AVVISO] Nessun oggetto JSON trovato nella risposta da '{provider}'.")
            return None
        open_braces = 0
        json_end = -1
        for i, char in enumerate(single_line_response[json_start:]):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            if open_braces == 0:
                json_end = json_start + i + 1
                break
        if json_end == -1:
            print(f"\n[AVVISO] Nessun oggetto JSON completo trovato nella risposta da '{provider}'.")
            return None
        clean_json_str = single_line_response[json_start:json_end]
        try:
            parsed_json = json.loads(clean_json_str)
        except json.JSONDecodeError:
            print(f"\n[AVVISO] Errore di decodifica JSON dalla risposta di '{provider}'. Risposta: {clean_json_str}")
            return None
        if not all(k in parsed_json for k in ["summary", "tags", "title"]):
            print(f"\n[AVVISO] JSON da '{provider}' è incompleto. Risposta: {clean_json_str}. Salto articolo.")
            return None
        return parsed_json
    except requests.exceptions.RequestException as e:
        print(f"\n[ERRORE FATALE] Errore di connessione o timeout con {provider.upper()}: {e}")
        print(f"Il processo di analisi verrà interrotto. Controlla che il servizio sia in esecuzione e non sovraccarico.")
        raise
    except Exception as e:
        print(f"\n[ERRORE IMPREVISTO] in process_with_llm: {e}")
        return None

def process_article(entry, api_url, llm_provider, model_name, generate_audio_flag=False):
    original_title = entry.get('title', 'No Title')
    link = entry.get('link', '')
    if 'news.ycombinator.com/item?id=' in link:
        hn_id_raw = urlparse(link).query.replace('id=', '')
        try:
            response = requests.get(link, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            title_span = soup.select_one('.titleline > a')
            if title_span and title_span.get('href'):
                link = title_span['href']
        except Exception as e:
            print(f"Errore nel recupero del link esterno da {link}: {e}")
    else:
        hn_id_raw = link.split('/')[-1]
    hn_id = sanitize_filename(hn_id_raw)
    if not hn_id:
        hn_id = hashlib.md5(original_title.encode()).hexdigest()[:10]
    domain = urlparse(link).netloc
    text = extract_text_from_url(link)
    if "Errore" in text:
        print(f'Salto articolo "{original_title}" a causa di un errore di estrazione: {text}')
        return None
    result = process_with_llm(text, link, api_url, llm_provider, model_name)
    if result is None:
        print(f'Salto articolo "{original_title}" a causa di un errore di analisi LLM.')
        return None
    tags = result.get("tags", [])
    macro_tags = get_macro_tags(tags)
    processed_article = {
        "hn_id": hn_id,
        "title": result.get("title", original_title),
        "original_title": original_title,
        "link": link,
        "domain": domain,
        "summary": result.get("summary", "Nessun riassunto disponibile."),
        "tags": tags,
        "macro_tags": macro_tags,
        "main_link": result.get("main_link", "")
    }
    if generate_audio_flag and processed_article["summary"]:
        safe_audio_name = f"{hn_id[:50]}.mp3"
        audio_path = generate_audio(processed_article["summary"], safe_audio_name, processed_article["title"])
        if audio_path:
            processed_article["audio_file"] = os.path.basename(audio_path)
            print(f"Audio generato: {processed_article['audio_file']}")
    if link and not any(d in domain for d in ['youtube.com', 'reddit.com', 'twitter.com']):
        safe_screenshot_name = f"{hn_id[:50]}.png"
        screenshot_path, thumbnail_base64 = capture_screenshot(link, safe_screenshot_name)
        if screenshot_path:
            processed_article["screenshot_file"] = os.path.basename(screenshot_path)
            processed_article["thumbnail_base64"] = thumbnail_base64
            print(f"Screenshot catturato: {processed_article['screenshot_file']}")
    filename = os.path.join(OUTPUT_DIR, f"{hn_id}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(processed_article, f, ensure_ascii=False, indent=2)
    return processed_article

def process_all_articles(entries, config, limit=None, generate_audio=False, workers=2):
    if limit:
        entries = entries[:limit]
    results = []
    print(f"Analisi di {len(entries)} articoli in corso (max {workers} worker)... con {config['llm_provider']} su {config['api_url']} usando il modello {config['model_name']}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        from functools import partial
        process_func = partial(process_article, 
                               api_url=config['api_url'], 
                               llm_provider=config['llm_provider'],
                               model_name=config['model_name'],
                               generate_audio_flag=generate_audio)
        future_to_entry = {executor.submit(process_func, entry): entry for entry in entries}
        for future in tqdm(future_to_entry, total=len(entries)):
            entry = future_to_entry[future]
            original_title = entry.get('title', 'No Title')
            try:
                result = future.result()
                if result:
                    results.append(result)
            except requests.exceptions.RequestException:
                print(f"\n[ERRORE FATALE] Rilevato errore di connessione in un worker. Arresto del processo.")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:
                print(f"\n[ERRORE WORKER] Errore durante l'analisi di '{original_title}': {e}. L'analisi continua.")
    with open(os.path.join(OUTPUT_DIR, "all_articles.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

def generate_html(articles):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(OUTPUT_DIR, f"hn_analysis_{timestamp}.html")
    template_path = os.path.join(os.path.dirname(__file__), "hn_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    articles_html = ""
    all_summaries = []
    for i, article in enumerate(articles):
        num = i + 1
        title = article.get("title", "Titolo non disponibile")
        link = article.get("link", "#")
        domain = article.get("domain", "")
        summary = article.get("summary", "Riassunto non disponibile")
        tags = article.get("tags", [])
        macro_tags = article.get("macro_tags", [])
        main_link = article.get("main_link", "")
        audio_file = article.get("audio_file", "")
        screenshot_file = article.get("screenshot_file", "")
        thumbnail_base64 = article.get("thumbnail_base64", "")
        article_id = f"article-{i}"
        all_summaries.append(f"Articolo {num}. {title}. {summary}")
        screenshot_html = ""
        if thumbnail_base64:
            onclick_handler = f"openModal('{article_id}', '{thumbnail_base64}')"
            screenshot_html = f'<img src="{thumbnail_base64}" alt="Screenshot di {title}" class="screenshot-thumb" onclick="{onclick_handler}">'
        articles_html += f'''
        <article class="article" id="{article_id}">
            <div class="article-content-wrapper">
                <div class="article-image-col">
                    {screenshot_html if screenshot_html else ''}
                </div>
                <div class="article-main-col">
                    <div class="article-body">
                        <div style="font-size:1.1em;font-weight:700;color:#8f5cff;margin-bottom:2px;">{num}</div>
                        <h2><a href="{link}" target="_blank">{title}</a></h2>
                        <div class="domain">{domain}</div>
                        {('<div class="macro-tags">' + ' '.join([f'<span class="macro-tag">{mt}</span>' for mt in macro_tags]) + '</div>') if macro_tags else ''}
                        <div class="summary">{summary}</div>
                        <div class="tags">
                            {' '.join([f'<span class="tag">{tag}</span>' for tag in tags])}
                        </div>
                        {f'<div class="main-link">Link principale: <a href="{main_link}" target="_blank">{main_link}</a></div>' if main_link else ''}
                        {f'<a href="../{AUDIO_DIR}/{audio_file}" class="download-audio" download>📥 Scarica Audio MP3</a>' if audio_file else ''}
                    </div>
                    <div class="article-actions">
                        <div class="audio-controls">
                            <button class="audio-btn" onclick="speakText(document.querySelector('#{article_id} .summary').textContent, this)">
                                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"></path></svg>
                                Ascolta
                            </button>
                            <button class="audio-btn-play-from" onclick="playFromIndex({i})">
                                ▶️ Da qui
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </article>
        '''
    filter_info = "Tutti gli articoli"
    html_out = template.replace("{{date}}", time.strftime("%d/%m/%Y")) \
        .replace("{{filter_info}}", filter_info) \
        .replace("{{articles_html}}", articles_html) \
        .replace("{{all_summaries}}", "\n---\n".join(all_summaries)) \
        .replace("{{datetime}}", time.strftime("%d/%m/%Y alle %H:%M:%S")) \
        .replace("{{num_articles}}", str(len(articles)))
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_out)
    return filename

def cleanup_old_data(days_to_keep=7):
    now = datetime.now()
    cutoff = now - timedelta(days=days_to_keep)
    print(f"\nPulizia dei file più vecchi di {days_to_keep} giorni...")
    for dir_name in [OUTPUT_DIR, AUDIO_DIR, SCREENSHOT_DIR]:
        if not os.path.exists(dir_name): continue
        for file_name in os.listdir(dir_name):
            file_path = os.path.join(dir_name, file_name)
            if os.path.isfile(file_path):
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff:
                        os.remove(file_path)
                        print(f"Eliminato: {file_path}")
                except Exception as e:
                    print(f"Errore durante l'eliminazione di {file_path}: {e}")

def check_llm_status(api_url, provider):
    try:
        if provider == "ollama":
            check_url = urlparse(api_url)._replace(path="", query="", fragment="").geturl()
            print(f"Verifica dello stato di Ollama a: {check_url}...")
            response = requests.get(check_url, timeout=5)
            if response.status_code == 200 and "Ollama is running" in response.text:
                print("Servizio Ollama rilevato e funzionante.")
                return True
            else:
                print(f"\n[ERRORE] Ollama ha risposto ma potrebbe non essere operativo (Status: {response.status_code}).")
                return False
        elif provider == "lmstudio":
            check_url = urlparse(api_url)._replace(path="/v1/models").geturl()
            print(f"Verifica dello stato di LM Studio a: {check_url}...")
            response = requests.get(check_url, timeout=10)
            if response.status_code == 200:
                print("Servizio LM Studio (o compatibile OpenAI) rilevato e funzionante.")
                return True
            else:
                print(f"\n[ERRORE] LM Studio ha risposto ma potrebbe non essere operativo (Status: {response.status_code}).")
                return False
    except requests.exceptions.RequestException:
        print(f"\n[ERRORE] Impossibile connettersi a {provider.upper()} all'URL: {api_url}")
        print(f"Assicurati che il servizio sia in esecuzione e l'URL sia corretto.")
        return False
    return False

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description='Analizza il feed di Hacker News usando un LLM locale.')
    parser.add_argument('--limit', type=int, default=None, help='Numero massimo di articoli da analizzare.')
    parser.add_argument('--generate-audio', action='store_true', help='Genera file audio MP3 per i riassunti.')
    parser.add_argument('--workers', type=int, default=2, help='Numero di worker paralleli per l\'analisi (default: 2).')
    parser.add_argument('--configure', action='store_true', help='Forza la configurazione interattiva dell\'LLM.')
    parser.add_argument('--cleanup', type=int, default=None, metavar='DAYS', help='Pulisce i dati più vecchi del numero di giorni specificato.')
    args = parser.parse_args()

    config = get_llm_config()
    if args.configure or not config:
        config = interactive_config(config)

    if not check_llm_status(config['api_url'], config['llm_provider']):
        sys.exit(1)

    setup_environment()

    if args.cleanup is not None:
        cleanup_old_data(days_to_keep=args.cleanup)
        sys.exit(0)

    # Controlla se esiste un report recente
    latest_report_path = None
    if os.path.exists(OUTPUT_DIR):
        list_of_reports = glob.glob(os.path.join(OUTPUT_DIR, 'hn_analysis_*.html'))
        if list_of_reports:
            latest_report_path = max(list_of_reports, key=os.path.getmtime)

    if latest_report_path:
        report_time = datetime.fromtimestamp(os.path.getmtime(latest_report_path))
        if datetime.now() - report_time < timedelta(hours=6):
            print(f"\nL'ultimo report generato ({os.path.basename(latest_report_path)}) ha meno di 6 ore.")
            choice = ""
            while choice not in ['a', 'n']:
                choice = input("Vuoi aprirlo [a] o generarne uno nuovo [n]? ").lower().strip()
            
            if choice == 'a':
                print(f"Apertura del report esistente: {latest_report_path}")
                try:
                    webbrowser.open(f"file://{os.path.abspath(latest_report_path)}")
                except Exception as e:
                    print(f"Impossibile aprire il browser automaticamente: {e}")
                    print(f"Apri manualmente il file: {os.path.abspath(latest_report_path)}")
                sys.exit(0)

    entries = fetch_hackernews_feed()
    if not entries:
        print("Nessun articolo trovato. Uscita.")
        return

    articles = []
    try:
        articles = process_all_articles(
            entries, 
            config,
            args.limit, 
            args.generate_audio, 
            args.workers
        )
    except Exception:
        print(f"\n[ERRORE FATALE] Il processo di analisi è stato interrotto a causa di un errore in un worker.")
        sys.exit(1)

    if not articles:
        print("\nNessun articolo è stato processato correttamente. Uscita.")
        return

    html_file = generate_html(articles)
    print(f"\nAnalisi completata! Generata la pagina HTML: {html_file}")
    try:
        webbrowser.open(f"file://{os.path.abspath(html_file)}")
    except Exception as e:
        print(f"Impossibile aprire il browser automaticamente: {e}")
        print(f"Apri manualmente il file: {os.path.abspath(html_file)}")

if __name__ == "__main__":
    main()
