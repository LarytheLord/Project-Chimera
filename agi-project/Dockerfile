# Base image with Python 3.11
FROM python:3.11-slim

# Install Rust for performance-critical modules
RUN apt-get update && apt-get install -y curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Install Poetry for dependency management
RUN pip install poetry

# Copy dependency definitions
COPY pyproject.toml poetry.lock* ./

# Install dependencies
# --no-root is important to avoid installing the project itself, just the deps
RUN poetry install --no-root

# Copy the rest of the application's code
COPY . .

# Command to run the AGI (placeholder for now)
CMD ["poetry", "run", "python", "src/main.py"]
