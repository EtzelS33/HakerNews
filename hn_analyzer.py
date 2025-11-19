import requests
import feedparser
import json
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
from concurrent.futures import ThreadPoolExecutor
import argparse
from tqdm import tqdm
import webbrowser
from gtts import gTTS
from playwright.sync_api import sync_playwright
import hashlib
import glob
from datetime import datetime, timedelta
import base64
from PIL import Image
import io

# Configurazioni
OLLAMA_API_URL = "http://172.25.48.1:11434/api/generate"
MODEL_NAME = "gemma3:latest"
OUTPUT_DIR = "hn_analysis"
AUDIO_DIR = "hn_audio"
SCREENSHOT_DIR = "hn_screenshots"
MAX_WORKERS = 2  # Numero di thread paralleli (ridotto per evitare sovraccarico)

# Definizione macro-tag
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

def get_macro_tags(tags):
    """Mappa i tag agli appropriati macro-tag."""
    macro_tags_found = set()
    
    for tag in tags:
        tag_lower = tag.lower()
        for macro_tag, keywords in MACRO_TAGS.items():
            # Controlla se il tag corrisponde esattamente o è contenuto in una keyword
            for keyword in keywords:
                if tag_lower == keyword.lower() or keyword.lower() in tag_lower or tag_lower in keyword.lower():
                    macro_tags_found.add(macro_tag)
                    break
    
    return list(macro_tags_found)

def generate_audio(text, filename, title=""):
    """Genera un file audio MP3 dal testo usando gTTS."""
    try:
        # Prepara il testo completo da leggere
        if title:
            full_text = f"{title}. {text}"
        else:
            full_text = text
            
        # Crea oggetto gTTS
        tts = gTTS(text=full_text, lang='it', slow=False)
        
        # Salva come MP3
        audio_path = os.path.join(AUDIO_DIR, filename)
        tts.save(audio_path)
        
        return audio_path
    except Exception as e:
        print(f"⚠️ Errore nella generazione audio per {filename}: {e}")
        return None

def capture_screenshot(url, filename):
    """Cattura uno screenshot del sito web e crea miniatura base64."""
    try:
        with sync_playwright() as p:
            # Usa chromium in modalità headless
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            
            # Imposta timeout e naviga
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Aspetta un momento per il caricamento completo
            page.wait_for_timeout(2000)
            
            # Salva screenshot
            screenshot_path = os.path.join(SCREENSHOT_DIR, filename)
            page.screenshot(path=screenshot_path, full_page=False)
            
            browser.close()
            
            # Crea miniatura e converti in base64
            thumbnail_base64 = create_thumbnail_base64(screenshot_path)
            
            return screenshot_path, thumbnail_base64
    except Exception as e:
        print(f"⚠️ Errore nella cattura screenshot per {url}: {e}")
        return None, None

def create_thumbnail_base64(image_path, max_width=300):
    """Crea una miniatura dell'immagine e la converte in base64."""
    try:
        # Apri l'immagine
        with Image.open(image_path) as img:
            # Calcola le dimensioni della miniatura mantenendo le proporzioni
            width, height = img.size
            aspect_ratio = height / width
            new_width = min(width, max_width)
            new_height = int(new_width * aspect_ratio)
            
            # Ridimensiona l'immagine
            thumbnail = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Salva in buffer come JPEG per compressione
            buffer = io.BytesIO()
            # Converti RGBA in RGB se necessario
            if thumbnail.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', thumbnail.size, (255, 255, 255))
                background.paste(thumbnail, mask=thumbnail.split()[-1] if thumbnail.mode == 'RGBA' else None)
                thumbnail = background
            
            thumbnail.save(buffer, format='JPEG', quality=85, optimize=True)
            
            # Converti in base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"⚠️ Errore nella creazione della miniatura: {e}")
        return None

def setup_environment():
    """Prepara l'ambiente di lavoro creando le cartelle necessarie."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
    
    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)
    
    print(f"🔧 Ambiente configurato. I risultati saranno salvati in: {OUTPUT_DIR}")
    print(f"🎵 I file audio saranno salvati in: {AUDIO_DIR}")
    print(f"📸 Gli screenshot saranno salvati in: {SCREENSHOT_DIR}")

def fetch_hackernews_feed(url="https://news.ycombinator.com/rss"):
    """Recupera il feed RSS di Hacker News."""
    try:
        feed = feedparser.parse(url)
        print(f"📰 Recuperati {len(feed.entries)} articoli dal feed di Hacker News")
        return feed.entries
    except Exception as e:
        print(f"⚠️ Errore nel recupero del feed: {e}")
        return []

def extract_text_from_url(url):
    """Estrae il contenuto testuale da una URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return f"Errore: Status code {response.status_code}"
        
        # Usa BeautifulSoup per estrarre il testo
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Rimuove script e stili
        for script in soup(["script", "style"]):
            script.extract()
        
        # Estrae testo e normalizza spazi
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limita la lunghezza per evitare payload troppo grandi
        return text[:15000]  # Limita a ~15K caratteri
    except Exception as e:
        return f"Errore nell'estrazione del testo: {str(e)}"

def process_with_ollama(text, url):
    """Processa il testo con Ollama Gemma per ottenere riassunto e tag."""
    try:
        # Costruisci un prompt ben strutturato per Gemma
        prompt = f"""
        Analizza il seguente testo proveniente dall'URL: {url}

        TESTO:
        {text}

        Restituisci l'output in formato JSON con questa struttura:
        {{
            "title": "Titolo tradotto in italiano",
            "summary": "Un riassunto IN ITALIANO conciso (circa 100 parole) del contenuto. IMPORTANTE: Traduci tutto in italiano",
            "tags": ["tag1", "tag2", "tag3", ...],
            "main_link": "URL principale menzionata nell'articolo o vuoto se non presente"
        }}

        IMPORTANTE: Sia il titolo che il riassunto DEVONO essere in italiano, non in inglese.
        Assicurati che i tag siano rilevanti, specifici e in numero da 3 a 7.
        """

        # Chiamata API a Ollama
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=180  # Aumentato a 3 minuti
        )
        
        if response.status_code != 200:
            return {
                "summary": f"Errore nella chiamata Ollama: Status code {response.status_code}",
                "tags": ["error", "api_failure"],
                "main_link": ""
            }
        
        # Estrai la risposta e individua il JSON
        result = response.json()
        response_text = result.get('response', '')
        
        # Cerca di estrarre il JSON dalla risposta
        try:
            # Trova l'inizio e la fine del JSON nella risposta
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > 0:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Verifica che il JSON contenga i campi attesi
                if not all(k in parsed for k in ["summary", "tags"]):
                    raise ValueError("JSON incompleto")
                
                return parsed
            else:
                raise ValueError("Formato JSON non trovato")
                
        except Exception as e:
            # Se non riusciamo a estrarre JSON, creiamo una struttura manualmente
            # cercando di individuare le parti nella risposta testuale
            fallback = {
                "summary": "Non è stato possibile estrarre un riassunto strutturato.",
                "tags": ["parsing_error"],
                "main_link": ""
            }
            
            # Tenta di trovare un riassunto
            if "summary" in response_text.lower():
                summary_start = response_text.lower().find("summary")
                summary_end = response_text.lower().find("tags")
                if summary_start > 0 and summary_end > summary_start:
                    fallback["summary"] = response_text[summary_start:summary_end].strip()
            
            # Tenta di trovare i tag
            if "tags" in response_text.lower():
                tags_section = response_text.lower().split("tags")[1]
                potential_tags = [t.strip() for t in tags_section.split(',')][:5]
                if potential_tags:
                    fallback["tags"] = [t.strip('"[]') for t in potential_tags if t]
            
            return fallback
            
    except Exception as e:
        return {
            "summary": f"Errore nel processing: {str(e)}",
            "tags": ["processing_error"],
            "main_link": ""
        }

def process_article(entry, generate_audio=False):
    """Processa un singolo articolo."""
    original_title = entry.get('title', 'No Title')
    link = entry.get('link', '')
    hn_id = urlparse(link).path.split('/')[-1] if 'item?id=' in link else link.split('/')[-1]
    
    # Recupera il link effettivo dell'articolo da HN
    if 'news.ycombinator.com' in link:
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Cerca il link principale nella pagina di HN
            title_span = soup.select_one('.titleline')
            if title_span and title_span.a:
                link = title_span.a['href']
        except Exception as e:
            print(f"⚠️ Errore nel recupero del link dall'articolo HN: {e}")
    
    # Estrai il dominio dall'URL
    domain = urlparse(link).netloc
    
    # Estrai il testo dalla pagina
    text = extract_text_from_url(link)
    
    # Processa il testo con Ollama
    result = process_with_ollama(text, link)
    
    # Ottieni i tag dal risultato
    tags = result.get("tags", [])
    
    # Mappa i tag ai macro-tag
    macro_tags = get_macro_tags(tags)
    
    # Usa il titolo tradotto se disponibile, altrimenti usa l'originale
    translated_title = result.get("title", original_title)
    summary = result.get("summary", "Nessun riassunto disponibile")
    
    # Genera audio MP3 se il riassunto è disponibile e se richiesto
    audio_filename = None
    if generate_audio and summary and summary != "Nessun riassunto disponibile" and "Errore" not in summary:
        safe_filename = f"{hn_id[:50]}.mp3" if hn_id else f"{hash(original_title) % 10000}.mp3"
        audio_path = generate_audio(summary, safe_filename, translated_title)
        if audio_path:
            audio_filename = safe_filename
            print(f"🎵 Audio generato: {audio_filename}")
    
    # Cattura screenshot del sito se non è YouTube o domini problematici
    screenshot_filename = None
    thumbnail_base64 = None
    if link and not any(domain in link for domain in ['youtube.com', 'youtu.be', 'reddit.com']):
        safe_screenshot_name = f"{hn_id[:50]}.png" if hn_id else f"{hash(original_title) % 10000}.png"
        screenshot_path, thumbnail_base64 = capture_screenshot(link, safe_screenshot_name)
        if screenshot_path:
            screenshot_filename = safe_screenshot_name
            print(f"📸 Screenshot catturato: {screenshot_filename}")
    
    # Costruisci il risultato
    processed_article = {
        "hn_id": hn_id,
        "title": translated_title,
        "original_title": original_title,
        "link": link,
        "domain": domain,
        "summary": summary,
        "tags": tags,
        "macro_tags": macro_tags,
        "main_link": result.get("main_link", ""),
        "audio_file": audio_filename,
        "screenshot_file": screenshot_filename,
        "thumbnail_base64": thumbnail_base64
    }
    
    # Salva i risultati in un file JSON
    filename = f"{OUTPUT_DIR}/{hn_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(processed_article, f, ensure_ascii=False, indent=2)
    
    return processed_article

def process_all_articles(entries, limit=None, generate_audio=False):
    """Processa tutti gli articoli con multithreading."""
    if limit:
        entries = entries[:limit]
    
    results = []
    all_tags = set()
    all_macro_tags = set()
    
    print(f"🔍 Analisi di {len(entries)} articoli in corso...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Crea una funzione parziale con generate_audio
        from functools import partial
        process_func = partial(process_article, generate_audio=generate_audio)
        
        for result in tqdm(executor.map(process_func, entries), total=len(entries)):
            results.append(result)
            all_tags.update(result.get("tags", []))
            all_macro_tags.update(result.get("macro_tags", []))
    
    # Salva i risultati aggregati
    with open(f"{OUTPUT_DIR}/all_articles.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results, list(all_tags), list(all_macro_tags)

def filter_by_tags(articles, selected_tags):
    """Filtra gli articoli in base ai tag selezionati."""
    if not selected_tags:
        return articles
    
    filtered = []
    for article in articles:
        article_tags = set(article.get("tags", []))
        if any(tag in article_tags for tag in selected_tags):
            filtered.append(article)
    
    return filtered

def generate_html(articles, selected_tags):
    """Genera una pagina HTML con i risultati."""

    # Nuova versione: usa template HTML
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{OUTPUT_DIR}/hn_analysis_{timestamp}.html"
    template_path = os.path.join(os.path.dirname(__file__), "hn_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Genera HTML per tutti gli articoli
    articles_html = ""
    all_summaries = []
    for i, article in enumerate(articles):
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
        all_summaries.append(f"{title}. {summary}")
        screenshot_html = ""
        if thumbnail_base64 and screenshot_file:
            onclick_handler = f"openModal('{article_id}', '../{SCREENSHOT_DIR}/{screenshot_file}')"
            screenshot_html = f'<img src="{thumbnail_base64}" alt="Screenshot di {title}" class="screenshot-thumb" onclick="{onclick_handler}">'
        articles_html += f'''
        <article class="article" id="{article_id}">
            <button class="audio-btn" onclick="speakText(document.querySelector('#{article_id} .summary').textContent, this)">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"></path>
                </svg>
                Ascolta
            </button>
            <h2><a href="{link}" target="_blank">{title}</a></h2>
            <div class="domain">{domain}</div>
            {('<div class="macro-tags">' + ' '.join([f'<span class="macro-tag">{mt}</span>' for mt in macro_tags]) + '</div>') if macro_tags else ''}
            <div class="summary">{summary}</div>
            <div class="tags">
                {' '.join([f'<span class="tag">{tag}</span>' for tag in tags])}
            </div>
            {f'<div class="main-link">Link principale: <a href="{main_link}" target="_blank">{main_link}</a></div>' if main_link else ''}
            {f'<a href="../{AUDIO_DIR}/{audio_file}" class="download-audio" download>📥 Scarica Audio MP3</a>' if audio_file else ''}
            {screenshot_html}
        </article>
        '''

    # Prepara i dati per il template
    filter_info = f"Filtrato per tag: {', '.join(selected_tags)}" if selected_tags else "Tutti gli articoli"
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
    """Pulisce i vecchi dati (file più vecchi di days_to_keep giorni)."""
    try:
        now = datetime.now()
        cutoff_date = now - timedelta(days=days_to_keep)
        
        files_deleted = 0
        total_size_freed = 0
        
        # Pulisci i file JSON di analisi
        json_files = glob.glob(f"{OUTPUT_DIR}/*.json")
        for file_path in json_files:
            if os.path.exists(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    files_deleted += 1
                    total_size_freed += file_size
        
        # Pulisci i file audio
        audio_files = glob.glob(f"{AUDIO_DIR}/*.mp3")
        for file_path in audio_files:
            if os.path.exists(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    files_deleted += 1
                    total_size_freed += file_size
        
        # Pulisci gli screenshot
        screenshot_files = glob.glob(f"{SCREENSHOT_DIR}/*.png")
        for file_path in screenshot_files:
            if os.path.exists(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    files_deleted += 1
                    total_size_freed += file_size
        
        # Pulisci i file HTML
        html_files = glob.glob(f"{OUTPUT_DIR}/hn_analysis_*.html")
        for file_path in html_files:
            if os.path.exists(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    files_deleted += 1
                    total_size_freed += file_size
        
        # Converti la dimensione in formato leggibile
        if total_size_freed > 1024 * 1024:
            size_str = f"{total_size_freed / (1024 * 1024):.2f} MB"
        elif total_size_freed > 1024:
            size_str = f"{total_size_freed / 1024:.2f} KB"
        else:
            size_str = f"{total_size_freed} bytes"
        
        return files_deleted, size_str
        
    except Exception as e:
        print(f"⚠️ Errore durante la pulizia: {e}")
        return 0, "0 bytes"

def main():
    # Configurazione parser argomenti
    parser = argparse.ArgumentParser(description='Analizza il feed di Hacker News usando Ollama Gemma')
    parser.add_argument('--limit', type=int, default=None, help='Numero massimo di articoli da analizzare')
    parser.add_argument('--force', action='store_true', help='Forza la ri-analisi anche se i dati esistono già')
    parser.add_argument('--generate-audio', action='store_true', help='Genera file audio MP3 per i riassunti (default: disabilitato)')
    args = parser.parse_args()
    
    # Setup dell'ambiente
    setup_environment()
    
    # Mostra stato generazione audio
    if args.generate_audio:
        print("🎵 Generazione audio MP3 abilitata")
    else:
        print("🔇 Generazione audio MP3 disabilitata (usa solo sintesi vocale HTML)")
    
    # Recupera il feed di Hacker News
    entries = fetch_hackernews_feed()
    if not entries:
        print("❌ Nessun articolo trovato. Uscita.")
        return
    
    # Processa tutti gli articoli
    articles, all_tags, all_macro_tags = process_all_articles(entries, args.limit, args.generate_audio)
    
    # Mostra i macro-tag disponibili
    print("\n🎯 Macro-tag disponibili:")
    for i, macro_tag in enumerate(sorted(all_macro_tags)):
        print(f"{i+1}. {macro_tag}")
    
    # Chiedi all'utente quali macro-tag interessano
    print("\n👉 Vuoi filtrare per macro-tag? (s/n):")
    use_macro = input("> ").strip().lower() == 's'
    
    selected_tags = []
    if use_macro:
        print("\n🎯 Inserisci i numeri dei macro-tag che ti interessano, separati da virgola (lascia vuoto per tutti):")
        selection = input("> ").strip()
        
        if selection:
            try:
                indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                sorted_macro_tags = sorted(all_macro_tags)
                selected_macro_tags = [sorted_macro_tags[idx] for idx in indices if 0 <= idx < len(sorted_macro_tags)]
                
                # Filtra articoli che hanno almeno uno dei macro-tag selezionati
                filtered_articles = []
                for article in articles:
                    article_macro_tags = article.get("macro_tags", [])
                    if any(mt in article_macro_tags for mt in selected_macro_tags):
                        filtered_articles.append(article)
                        
                print(f"\n✅ Filtrati {len(filtered_articles)} articoli con i macro-tag selezionati")
            except:
                print("⚠️ Selezione non valida, verranno mostrati tutti gli articoli.")
                filtered_articles = articles
        else:
            filtered_articles = articles
    else:
        # Mostra tutti i tag disponibili
        print("\n🏷️ Tag trovati:")
        for i, tag in enumerate(sorted(all_tags)):
            print(f"{i+1}. {tag}")
        
        # Chiedi all'utente quali tag interessano
        print("\n👉 Inserisci i numeri dei tag che ti interessano, separati da virgola (lascia vuoto per tutti):")
        selection = input("> ").strip()
        
        if selection:
            try:
                indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                sorted_tags = sorted(all_tags)
                selected_tags = [sorted_tags[idx] for idx in indices if 0 <= idx < len(sorted_tags)]
            except:
                print("⚠️ Selezione non valida, verranno mostrati tutti gli articoli.")
        
        # Filtra gli articoli in base ai tag
        filtered_articles = filter_by_tags(articles, selected_tags)
    
    # Genera la pagina HTML
    html_file = generate_html(filtered_articles, selected_tags)
    
    print(f"\n✅ Analisi completata! Generata la pagina HTML: {html_file}")
    
    # Apri il browser con la pagina generata
    webbrowser.open(f"file://{os.path.abspath(html_file)}")
    
    # Chiedi se pulire i vecchi dati
    print("\n🧹 Vuoi pulire i dati più vecchi di 7 giorni? (s/n):")
    cleanup_response = input("> ").strip().lower()
    
    if cleanup_response == 's':
        print("\n🗑️ Pulizia in corso...")
        files_deleted, size_freed = cleanup_old_data(days_to_keep=7)
        
        if files_deleted > 0:
            print(f"✅ Pulizia completata! Eliminati {files_deleted} file, liberati {size_freed}")
        else:
            print("✅ Nessun file vecchio da eliminare")
    else:
        print("🔒 Pulizia annullata, i dati sono stati conservati")

if __name__ == "__main__":
    main()
