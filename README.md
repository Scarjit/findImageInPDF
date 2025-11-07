# Find Image in PDF

Find which page(s) of a PDF contain a given image using perceptual hashing.

## Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Usage

### Web GUI

```bash
uv run python find_image_in_pdf_gui.py
```

### Command Line

```bash
uv run python find_image_in_pdf.py <pdf_file> <image_file>
```

