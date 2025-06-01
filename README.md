# OpenAI Documentation Crawler

A Python-based web crawler that automatically indexes OpenAI's documentation pages using Playwright for accurate SPA (Single Page Application) content extraction.

## Features

- SPA-aware crawling using Playwright
- Handles dynamic content loading
- Generates a structured JSON index of documentation URLs
- Respects URL patterns and filtering

## Prerequisites

- Python 3.7+
- Playwright

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install playwright
playwright install chromium
```

## Usage

Run the crawler:
```bash
python playwright_crawler.py
```

The script will:
1. Launch a headless Chromium browser
2. Crawl OpenAI's documentation pages
3. Save the results to `openai_docs_index.json`

## Output

The crawler generates a JSON file containing an array of documentation URLs, structured and sorted alphabetically.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- PumpkinEscobar 