# Use Python 3.11 slim image as base
FROM python:3.11-slim

# System dependencies for Playwright and basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Playwright dependencies
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xvfb \
    # Other utilities if strictly needed, otherwise remove
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-root user and group
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 --shell /bin/bash --create-home appuser

# Copy only the requirements file first to leverage Docker cache
COPY --chown=appuser:appgroup requirements-secure.txt .

# Install Python dependencies as the appuser
# This step might need root if playwright install-deps needs it, 
# but pip install itself can run as non-root if /app is writable by appuser or a virtual env is used.
# For playwright install-deps, it's better to run it as root BEFORE switching user if it requires system-wide changes.

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install playwright system dependencies as root first
RUN playwright install-deps chromium

# Install Python dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements-secure.txt
RUN pip install playwright # Playwright Python package

# Copy the rest of the application as the appuser
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose the port the app runs on (must be >1024 for non-root)
EXPOSE 5000

# Healthcheck (Optional but recommended)
# HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
# CMD curl -f http://localhost:5000/health || exit 1

# Command to run the API
CMD ["python", "wizard_api.py"] 