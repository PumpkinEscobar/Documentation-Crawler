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

- Python 3.8+
- Dependencies listed in `requirements.txt`
- OpenAI API key (for API documentation)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/PumpkinEscobar/Documentation-Crawler.git
cd Documentation-Crawler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY='your-api-key'  # Required for OpenAI API documentation
```

## Usage

Run the crawler:
```bash
python doc_crawler.py
```

The crawler will:
1. Crawl all configured documentation sources
2. Generate a structured knowledge map
3. Save results in:
   - `doc_index.yaml`: Structured documentation index
   - `knowledge_base.json`: Detailed documentation content
   - `crawler.log`: Crawling process logs

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 