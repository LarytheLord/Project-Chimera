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
COPY agi_project/pyproject.toml agi_project/poetry.lock* ./agi-project/

# Install dependencies only
RUN cd agi_project && poetry install --no-root

# Copy the rest of the application's code
COPY . .

# Install the project itself
RUN cd agi_project && poetry install

# Library image smoke check
CMD ["poetry", "run", "python", "-c", "import chimera; print('Project Chimera library image ready')"]
