# GEMINI.md - Project Analysis

## Project Overview

This project is a Python-based tool designed to analyze the Hacker News (HN) RSS feed. It automates the process of fetching top stories, extracting content from the linked articles, and using a locally-run Large Language Model (LLM) to generate insightful analysis.

The core functionality includes:
- Fetching the latest articles from the HN RSS feed.
- Extracting clean, readable text content from each article's URL.
- Interacting with a local LLM (supporting Ollama and LM Studio) to:
  - Translate titles into Italian.
  - Generate concise summaries in Italian.
  - Assign relevant tags and high-level "macro tags" (e.g., "🤖 AI & Machine Learning", "🔒 Sicurezza & Privacy").
- Optionally generating text-to-speech (TTS) audio files for each summary.
- Capturing screenshots of the article web pages for a visual preview.
- Compiling all the analyzed data into a single, interactive HTML report.

The final output is a user-friendly web page that presents the articles with their summaries, tags, audio playback, and screenshots, allowing for a quick and comprehensive overview of the latest tech news.

## Key Technologies

- **Backend:** Python 3
- **Frontend (Report):** HTML, CSS, JavaScript
- **Core Python Libraries:**
  - `requests`: For all HTTP communications.
  - `feedparser`: For parsing the RSS feed.
  - `BeautifulSoup4`: For HTML parsing and text extraction.
  - `gTTS`: For generating MP3 audio from text.
  - `Playwright`: For programmatic browser control to capture screenshots.
  - `Pillow`: For creating image thumbnails.
- **LLM Integration:** The script is configured to work with local LLM providers like **Ollama** and **LM Studio** (or any OpenAI-compatible API endpoint).

## Building and Running

### 1. Installation

**a. System Dependencies (for Debian/Ubuntu):**
The following command installs packages required for building Python dependencies and for Playwright.
```bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
```

**b. Python Dependencies:**
Install all required Python packages using pip.
```bash
pip install -r requirements.txt
```

**c. Browser Binaries for Playwright:**
Playwright needs to download browser binaries for capturing screenshots.
```bash
playwright install
```

### 2. Configuration

**a. LLM Service:**
Ensure your local LLM service (e.g., Ollama, LM Studio) is running.

**b. Script Configuration:**
The script will guide you through an interactive setup on its first run. You can also force this setup with the `--configure` flag.
```bash
# Run this to configure your LLM provider, API URL, and model
python hn_analyzer_unified.py --configure
```
This will create or update the `llm_settings.json` file with your preferences.

### 3. Execution

**a. Standard Run:**
To run the analysis with default settings, execute the main script.
```bash
python hn_analyzer_unified.py
```

**b. Run with Options:**
You can customize the execution with command-line arguments.
```bash
# Example: Analyze the top 5 articles, generate audio, and use 4 parallel workers
python hn_analyzer_unified.py --limit 5 --generate-audio --workers 4
```
- `--limit <N>`: Restricts analysis to the top `N` articles.
- `--generate-audio`: Creates an MP3 file for each summary.
- `--workers <N>`: Sets the number of articles to process in parallel.
- `--cleanup <DAYS>`: Deletes all generated data older than `DAYS`.

The script will automatically open the generated HTML report in your default web browser upon completion.

## Development Conventions

- The primary, cross-platform script is `hn_analyzer_unified.py`. Development should be focused here.
- The script is configured via `llm_settings.json`. Avoid hardcoding API keys or URLs.
- All generated output is stored in dedicated directories (`hn_analysis`, `hn_audio`, `hn_screenshots`) to keep the root directory clean.
- The HTML output is generated from the `hn_template.html` file, which can be modified to change the report's appearance.
- The code uses f-strings, basic type hinting, and includes error handling for network and LLM operations to ensure robustness.
