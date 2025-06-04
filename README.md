# AI Documentation Crawler

This project crawls and indexes documentation from major AI providers to create a comprehensive knowledge base for AI/LLM development, training, and deployment. The crawler is specifically designed to support a specialized chatbot focused on LLM training, prompting strategy, agent design, and operational deployment.

## Supported Documentation Sources

- OpenAI (API Reference, Documentation, Cookbook)
- Anthropic Documentation
- Google AI Developer Documentation
- Meta AI Resources

## Features

- Multi-source documentation crawling
- Structured knowledge mapping
- API documentation integration
- Comprehensive error handling and logging
- YAML-based documentation index
- JSON knowledge base output

## Requirements

- Docker (recommended) OR Python 3.8+
- Docker Compose (if using Docker)

## Setup

### Option 1: Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/PumpkinEscobar/Documentation-Crawler.git
cd Documentation-Crawler
```

2. Build and run using Docker Compose:
```bash
docker compose build
docker compose up
```

### Option 2: Local Setup

1. Clone the repository:
```bash
git clone https://github.com/PumpkinEscobar/Documentation-Crawler.git
cd Documentation-Crawler
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y python3-numpy python3-pandas python3-pip
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

5. Set up environment variables:
```bash
export OPENAI_API_KEY='your-api-key'  # Required for OpenAI API documentation
```

## Usage

### Using Docker:
```bash
docker compose up
```

### Using Local Setup:
```bash
python3 doc_crawler.py
```

The crawler will:
1. Crawl all configured documentation sources
2. Generate a structured knowledge map
3. Save results in:
   - `doc_index.yaml`: Structured documentation index
   - `knowledge_base.json`: Detailed documentation content
   - `crawler.log`: Crawling process logs

## Configuration

### crawler_config.json
The crawler can be configured using `crawler_config.json`:
```json
{
  "browser_options": {
    "args": ["--no-sandbox", "--disable-setuid-sandbox"]
  }
}
```

## Output Structure

### doc_index.yaml
```yaml
metadata:
  last_updated: "YYYY-MM-DD HH:MM:SS"
  sources: ["openai", "anthropic", "google_ai", "meta_ai"]
documentation:
  source_name:
    api_documentation:
      # API endpoints and models
    web_documentation:
      # Web-based documentation sections
    examples:
      # Code examples and tutorials
```

### knowledge_base.json
Contains the detailed content of all crawled documentation, including:
- API endpoints
- Model information
- Documentation sections
- Code examples
- Tutorials

## Error Handling

The crawler implements comprehensive error handling:
- Connection timeouts
- API rate limits
- Authentication failures
- Invalid URLs
- Content parsing errors

All errors are logged to `crawler.log` with appropriate context and timestamps.

## Troubleshooting

### Common Issues:

1. Playwright Sandbox Issues:
   - Ensure you're using the provided Docker setup or
   - Use the `--no-sandbox` configuration in `crawler_config.json`

2. Dependency Issues:
   - Use Docker setup to avoid dependency conflicts
   - If using local setup, ensure system packages are installed before pip packages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 