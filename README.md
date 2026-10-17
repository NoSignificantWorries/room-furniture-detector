# room-furniture-detector

Automatic furniture detector that extracts bounding boxes with class labels from room photos and outputs them as JSON. Built for real-estate systems to automate furniture description.

## Setup

### Web
```bash
uv sync --extra web

source ./.venv/bin/activate

cd web
npm install

# uv run pre-commit install
```

### Service
```bash
uv sync --extra serve

source ./.venv/bin/activate

# cd web
# npm install
```

## Train

## Production preparation

## Infer
