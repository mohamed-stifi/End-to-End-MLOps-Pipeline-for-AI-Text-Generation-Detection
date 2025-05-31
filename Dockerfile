# Base image
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -d ${APP_HOME} -s /sbin/nologin -c "Docker image user" appuser



# Install system dependencies
# git is needed for DVC (if pulling from git-based remotes) or if requirements.txt has git dependencies
# build-essential for C extensions that some Python packages might need
# wget/curl for downloading things if necessary (e.g., DVC binary if not pip installed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    # Add any other system dependencies your project might need
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Create app directory
WORKDIR ${APP_HOME}


# Install DVC (can be installed via pip too, but system package might be more stable for some remotes)
# Option 1: Using pip (simpler, will be part of requirements.txt)
# RUN pip install dvc[all] # or dvc[s3], dvc[gdrive], etc. depending on your remote

# Copy only essential files for dependency installation first
COPY requirements.txt setup.py README.md ./
# Copy the src directory which contains the actual package
COPY src/ ./src/

# Install dependencies, including the local package in editable mode
# This will find setup.py in the current WORKDIR (/app)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code needed at runtime
# (api, config, main.py, entrypoint.sh)
# .dockerignore will ensure large/unnecessary files from the root are not copied
COPY api/ ./api/
COPY config/ ./config/
COPY main.py .




# Download NLTK data (if not handled by the application itself on first run)
# This ensures the image has necessary data without runtime downloads for these specific packages.
# You can also create a script to run these and call it.
RUN python -m nltk.downloader stopwords punkt wordnet averaged_perceptron_tagger # Add others if needed by NLP preprocessing

# Create necessary directories that might be written to (if not volume-mounted)
# This is more for cases where you don't mount volumes, but generally, artifacts/logs are mounted.
RUN mkdir -p ${APP_HOME}/artifacts ${APP_HOME}/logs ${APP_HOME}/mlruns \
    && chown -R appuser:appuser ${APP_HOME}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Change to non-root user
USER appuser

# Expose port for the API
EXPOSE 8000

# Default command (can be overridden) - e.g., to start the API
# Use an entrypoint script for more flexibility (see below)
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Entrypoint script for flexibility (see entrypoint.sh example below)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"] # Default action for entrypoint: run api