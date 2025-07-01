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
import re
import sys

# --- CONFIGURATIONS ---
MODEL_NAME = "gemma3:latest"
OUTPUT_DIR = "hn_analysis"
AUDIO_DIR = "hn_audio"
SCREENSHOT_DIR = "hn_screenshots"

# --- MACRO TAGS ---
MACRO_TAGS = {
    "AI & Machine Learning": [
        "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
        "neural network", "llm", "large language model", "gpt", "generative ai",
        "computer vision", "nlp", "natural language processing", "transformer",
        "diffusion model", "stable diffusion", "midjourney", "dall-e", "chatgpt",
        "claude", "gemini", "copilot", "ai ethics", "agi", "reinforcement learning",
        "tensorflow", "pytorch", "scikit-learn", "hugging face"
    ],
    "Programmazione & DevOps": [
        "programming", "coding", "software development", "javascript", "python",
        "java", "c++", "rust", "go", "typescript", "react", "vue", "angular",
        "docker", "kubernetes", "devops", "ci/cd", "git", "github", "gitlab",
        "api", "microservices", "serverless", "cloud native", "agile", "scrum"
    ],
    "Sicurezza & Privacy": [
        "security", "cybersecurity", "privacy", "encryption", "cryptography",
        "vulnerability", "exploit", "hacking", "penetration testing", "firewall",
        "vpn", "tor", "gdpr", "data protection", "password", "authentication",
        "2fa", "zero trust", "ransomware", "malware", "phishing"
    ],
    "Web & Cloud": [
        "web development", "frontend", "backend", "full stack", "aws", "azure",
        "google cloud", "gcp", "cloud computing", "saas", "paas", "iaas",
        "cdn", "hosting", "domain", "ssl", "https", "rest api", "graphql",
        "websocket", "pwa", "jamstack", "edge computing"
    ],
    "Data & Database": [
        "database", "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        "elasticsearch", "data science", "big data", "data analytics", "etl",
        "data warehouse", "data lake", "hadoop", "spark", "kafka", "data visualization",
        "tableau", "power bi", "pandas", "numpy"
    ],
    "Startup & Business": [
        "startup", "entrepreneurship", "venture capital", "vc", "funding",
        "ipo", "acquisition", "merger", "business model", "saas business",
        "b2b", "b2c", "growth hacking", "product market fit", "unicorn",
        "y combinator", "accelerator", "pitch deck", "mvp", "lean startup"
    ],
    "Scienza & Ricerca": [
        "science", "research", "physics", "chemistry", "biology", "medicine",
        "quantum computing", "quantum physics", "astronomy", "space", "nasa",
        "climate change", "renewable energy", "biotechnology", "genetics",
        "crispr", "vaccine", "drug discovery", "clinical trial", "peer review"
    ],
    "Gaming & VR/AR": [
        "gaming", "video games", "game development", "unity", "unreal engine",
        "vr", "virtual reality", "ar", "augmented reality", "mr", "mixed reality",
        "metaverse", "oculus", "quest", "steamvr", "game design", "indie game",
        "aaa game", "mobile gaming", "esports"
    ],
    "Hardware & IoT": [
        "hardware", "cpu", "gpu", "semiconductor", "chip", "processor",
        "raspberry pi", "arduino", "iot", "internet of things", "embedded",
        "fpga", "asic", "sensor", "robotics", "3d printing", "maker",
        "diy electronics", "pcb", "firmware"
    ],
    "Crypto & Blockchain": [
        "cryptocurrency", "bitcoin", "ethereum", "blockchain", "defi",
        "nft", "web3", "smart contract", "dao", "mining", "wallet",
        "exchange", "stablecoin", "altcoin", "consensus", "proof of work",
        "proof of stake", "layer 2", "polygon", "solana"
    ]
}

# --- HELPER FUNCTIONS ---

def sanitize_filename(name):
    """Removes characters that are illegal in filenames."""
    name = name.split('?')[0]
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def get_macro_tags(tags):
    """Mappa i tag agli appropriati macro-tag."""
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
    """Genera un file audio MP3 dal testo usando gTTS."""
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
    """Cattura uno screenshot del sito web e crea miniatura base64."""
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
    """Crea una miniatura dell'immagine e la converte in base64."""
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
    """Prepara l'ambiente di lavoro creando le cartelle necessarie."""
    for directory in [OUTPUT_DIR, AUDIO_DIR, SCREENSHOT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print(f"Ambiente configurato. Output in: {OUTPUT_DIR}, {AUDIO_DIR}, {SCREENSHOT_DIR}")

def fetch_hackernews_feed(url="https://news.ycombinator.com/rss"):
    """Recupera il feed RSS di Hacker News."""
    try:
        feed = feedparser.parse(url)
        print(f"Recuperati {len(feed.entries)} articoli dal feed di Hacker News")
        return feed.entries
    except Exception as e:
        print(f"Errore nel recupero del feed: {e}")
        return []

def extract_text_from_url(url):
    """Estrae il contenuto testuale da una URL."""
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
        return text[:15000]
    except Exception as e:
        return f"Errore nell'estrazione del testo: {str(e)}"

# --- CORE OLLAMA & ARTICLE PROCESSING ---

def process_with_ollama(text, url, ollama_url):
    """Processa il testo con Ollama per ottenere riassunto e tag in formato JSON."""
    try:
        prompt = f"""
        Analizza il seguente testo proveniente dall'URL: {url}
        TESTO:
        {text}
        
        Restituisci l'output in formato JSON con questa esatta struttura:
        {{
            "title": "Titolo dell'articolo, tradotto in italiano",
            "summary": "Un riassunto conciso in ITALIANO (massimo 100 parole) del contenuto.",
            "tags": ["elenco", "di", "3-7", "tag", "rilevanti"],
            "main_link": "URL principale menzionata nell'articolo o stringa vuota se non presente"
        }}
        
        IMPORTANTE:
        - Traduci sempre il titolo e il riassunto in italiano.
        - I tag devono essere specifici e pertinenti.
        """
        
        response = requests.post(
            ollama_url,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "format": "json",  # Forza l'output JSON
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        
        result = response.json()
        parsed_json = json.loads(result.get('response', '{}'))

        if not all(k in parsed_json for k in ["summary", "tags", "title"]):
            raise ValueError("JSON ricevuto da Ollama è incompleto.")
            
        return parsed_json

    except requests.exceptions.RequestException as e:
        print(f"\n[ERRORE FATALE] Errore di connessione o timeout con Ollama: {e}")
        print("Il processo di analisi verrà interrotto. Controlla che Ollama sia in esecuzione e non sovraccarico.")
        raise
    except json.JSONDecodeError as e:
        print(f"\n[ERRORE FATALE] Fallimento nel parsing del JSON da Ollama: {e}")
        print(f"Risposta ricevuta: {result.get('response', '')}")
        raise
    except Exception as e:
        print(f"\n[ERRORE IMPREVISTO] in process_with_ollama: {e}")
        raise

def process_article(entry, ollama_url, generate_audio_flag=False):
    """Processa un singolo articolo: estrae, analizza, salva."""
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
            else:
                pass
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
        print(f"Salto articolo \"{original_title}\" a causa di un errore di estrazione: {text}")
        return None

    result = process_with_ollama(text, link, ollama_url)
    
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

def process_all_articles(entries, ollama_url, limit=None, generate_audio=False, workers=2):
    """Processa tutti gli articoli con multithreading."""
    if limit:
        entries = entries[:limit]
    
    results = []
    print(f"Analisi di {len(entries)} articoli in corso (max {workers} worker)...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        from functools import partial
        process_func = partial(process_article, ollama_url=ollama_url, generate_audio_flag=generate_audio)
        
        future_to_entry = {executor.submit(process_func, entry): entry for entry in entries}
        
        for future in tqdm(future_to_entry, total=len(entries)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    with open(os.path.join(OUTPUT_DIR, "all_articles.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

# --- HTML GENERATION & CLEANUP ---

def generate_html(articles):
    """Genera una pagina HTML con i risultati, includendo la funzionalità TTS."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(OUTPUT_DIR, f"hn_analysis_{timestamp}.html")
    
    html_start = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analisi Hacker News - {time.strftime("%d/%m/%Y")}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f9f9f9; }}
        header {{ background: #ff6600; color: white; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; }}
        h1 {{ margin: 0; font-size: 2em; }}
        .voice-selector-container {{ display: flex; align-items: center; gap: 10px; }}
        #voice-selector {{ padding: 8px; border-radius: 5px; border: 1px solid #ccc; font-size: 14px; background: white; }}
        .article {{ background: white; margin-bottom: 20px; padding: 25px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.08); display: flex; gap: 20px; }}
        .article-content {{ flex-grow: 1; }}
        .title-container {{ display: flex; align-items: center; gap: 15px; margin-bottom: 5px; }}
        .article h2 {{ margin: 0; }}
        .article h2 a {{ text-decoration: none; color: #0366d6; font-size: 1.4em; }}
        .article h2 a:hover {{ text-decoration: underline; }}
        .original-title {{ font-size: 0.9em; font-style: italic; color: #555; margin-top: 0; margin-bottom: 15px; }}
        .domain {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
        .tags, .macro-tags {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 15px; }}
        .tag {{ background: #f1f8ff; color: #0366d6; padding: 4px 12px; border-radius: 15px; font-size: 0.8em; border: 1px solid #cce5ff; }}
        .macro-tag {{ background: #ff6600; color: white; padding: 5px 14px; border-radius: 20px; font-size: 0.85em; font-weight: 500; }}
        .screenshot-container {{ flex-shrink: 0; width: 200px; }}
        .screenshot-thumb {{ width: 100%; border-radius: 5px; cursor: pointer; border: 1px solid #ddd; }}
        .audio-btn {{ background: #0366d6; color: white; border: none; padding: 6px 12px; border-radius: 5px; cursor: pointer; font-size: 12px; }}
        .audio-btn:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <header>
        <h1>Analisi Hacker News - {time.strftime("%d/%m/%Y")}</h1>
        <div class="voice-selector-container">
            <label for="voice-selector" style="color: white; font-weight: 500;">Voce TTS:</label>
            <select id="voice-selector"></select>
        </div>
    </header>
    <main>
    """

    html_content = []
    for i, article in enumerate(articles):
        article_id = f"article-{i}"
        screenshot_html = ""
        if article.get("thumbnail_base64"):
            screenshot_html = f"""
            <div class="screenshot-container">
                <a href="{article['link']}" target="_blank">
                    <img src="{article['thumbnail_base64']}" alt="Screenshot di {article['title']}" class="screenshot-thumb">
                </a>
            </div>"""

        article_html = f"""
        <article id="{article_id}">
            <div class="article-content">
                <div class="title-container">
                    <h2><a href="{article['link']}" target="_blank">{article['title']}</a></h2>
                    <button class="audio-btn" onclick="speakText(document.querySelector('#{article_id} .summary').textContent, this)">Ascolta</button>
                </div>
                <div class="original-title">{article['original_title']}</div>
                <div class="domain">{article['domain']}</div>
                <div class="summary">{article['summary']}</div>
                <div class="macro-tags">{' '.join(f'<span class="macro-tag">{mt}</span>' for mt in article['macro_tags'])}</div>
                <div class="tags">{' '.join(f'<span class="tag">{tag}</span>' for tag in article['tags'])}</div>
            </div>
            {screenshot_html}
        </article>
        """
        html_content.append(article_html)

    html_end = """
    </main>
    <script>
        const synth = window.speechSynthesis;
        const voiceSelector = document.getElementById('voice-selector');
        let voices = [];
        let currentlyPlaying = null; // Riferimento al pulsante dell'audio in riproduzione

        function populateVoiceList() {
            voices = synth.getVoices().filter(voice => voice.lang.startsWith('it'));
            const selectedIndex = voiceSelector.selectedIndex < 0 ? 0 : voiceSelector.selectedIndex;
            voiceSelector.innerHTML = '';
            if (voices.length === 0) {
                const option = document.createElement('option');
                option.textContent = 'Nessuna voce italiana trovata';
                voiceSelector.appendChild(option);
                return;
            }
            voices.forEach((voice, i) => {
                const option = document.createElement('option');
                option.textContent = `${voice.name} (${voice.lang})`;
                option.setAttribute('data-name', voice.name);
                voiceSelector.appendChild(option);
            });
            voiceSelector.selectedIndex = selectedIndex;
        }

        populateVoiceList();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = populateVoiceList;
        }

        function stopCurrentSpeech() {
            synth.cancel();
            if (currentlyPlaying) {
                currentlyPlaying.textContent = 'Ascolta';
                currentlyPlaying = null;
            }
        }

        function speakText(text, button) {
            if (synth.speaking && currentlyPlaying === button) {
                stopCurrentSpeech();
                return;
            }

            stopCurrentSpeech();
            
            const utterance = new SpeechSynthesisUtterance(text);
            const selectedVoiceName = voiceSelector.selectedOptions[0].getAttribute('data-name');
            const selectedVoice = voices.find(voice => voice.name === selectedVoiceName);
            
            if (selectedVoice) {
                utterance.voice = selectedVoice;
            }
            utterance.lang = 'it-IT';
            
            utterance.onend = () => {
                button.textContent = 'Ascolta';
                currentlyPlaying = null;
            };
            utterance.onerror = () => { 
                button.textContent = 'Ascolta';
                currentlyPlaying = null;
                alert('Errore nella sintesi vocale.');
            };
            
            synth.speak(utterance);
            button.textContent = 'Ferma';
            currentlyPlaying = button;
        }

        // Ferma la riproduzione quando si cambia voce
        voiceSelector.addEventListener('change', stopCurrentSpeech);
    </script>
</body></html>
"""
    full_html = html_start + "\n".join(html_content) + html_end
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    return filename

def cleanup_old_data(days_to_keep=7):
    """Pulisce i vecchi dati più vecchi di N giorni."""
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

def check_ollama_status(ollama_url):
    """Verifica se il servizio Ollama è in esecuzione."""
    try:
        base_url = urlparse(ollama_url)._replace(path="", query="", fragment="").geturl()
        print(f"Verifica dello stato di Ollama a: {base_url}...")
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200 and "Ollama is running" in response.text:
            print("Servizio Ollama rilevato e funzionante.")
            return True
        else:
            print(f"\n[ERRORE] Ollama ha risposto ma potrebbe non essere operativo (Status: {response.status_code}).")
            return False
    except requests.exceptions.RequestException:
        print("\n[ERRORE] Impossibile connettersi a Ollama.")
        print(f"Assicurati che Ollama sia in esecuzione e raggiungibile all'indirizzo specificato.")
        print("Puoi avviare Ollama dal menu Start o eseguendo 'ollama serve' nel terminale.")
        return False

# --- MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(description='Analizza il feed di Hacker News usando Ollama.')
    parser.add_argument('--limit', type=int, default=None, help='Numero massimo di articoli da analizzare.')
    parser.add_argument('--generate-audio', action='store_true', help='Genera file audio MP3 per i riassunti.')
    parser.add_argument('--workers', type=int, default=1, help='Numero di worker paralleli per l\'analisi (default: 2).')
    parser.add_argument('--ollama-url', type=str, default="http://127.0.0.1:11434/api/generate", help="URL dell'API di Ollama.")
    parser.add_argument('--cleanup', type=int, default=None, metavar='DAYS', help='Pulisce i dati più vecchi del numero di giorni specificato.')
    
    args = parser.parse_args()

    if not check_ollama_status(args.ollama_url):
        sys.exit(1)

    setup_environment()

    if args.cleanup is not None:
        cleanup_old_data(days_to_keep=args.cleanup)
        sys.exit(0)

    entries = fetch_hackernews_feed()
    if not entries:
        print("Nessun articolo trovato. Uscita.")
        return

    articles = []
    try:
        articles = process_all_articles(
            entries, 
            args.ollama_url, 
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