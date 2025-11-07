#!/usr/bin/env python3
"""
Find which page of a PDF contains a given image.
Works on both Windows and Linux.
"""

import sys
import argparse
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from PIL import Image
import imagehash


def calculate_perceptual_similarity(img1: Image.Image, img2: Image.Image, hash_size: int = 16) -> float:
    """
    Calculate similarity between two images using perceptual hashing.
    Returns a similarity score between 0 and 1 (1 being identical).

    Uses average hashing (aHash) which is fast and works well for finding similar images.
    """
    # Convert images to RGB mode to ensure consistent processing
    if img1.mode != 'RGB':
        img1 = img1.convert('RGB')
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')

    # Calculate perceptual hashes
    hash1 = imagehash.average_hash(img1, hash_size=hash_size)
    hash2 = imagehash.average_hash(img2, hash_size=hash_size)

    # Calculate Hamming distance (number of different bits)
    hamming_distance = hash1 - hash2

    # Convert to similarity score (0-1 range)
    # Maximum possible distance is hash_size^2 (total number of bits)
    max_distance = hash_size * hash_size
    similarity = 1.0 - (hamming_distance / max_distance)

    return similarity


def get_cache_path(pdf_path: Path) -> Path:
    """
    Get the cache file path for a given PDF.
    Cache key includes PDF name and modification time.
    """
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    # Create cache key from PDF path and modification time
    mtime = pdf_path.stat().st_mtime
    cache_key = f"{pdf_path.name}_{mtime}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    return cache_dir / f"{cache_hash}.pkl"


def load_cached_images(pdf_path: Path) -> Optional[Dict[int, List[Image.Image]]]:
    """
    Load cached extracted images for a PDF.
    Returns None if cache doesn't exist or is invalid.
    """
    cache_path = get_cache_path(pdf_path)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache: {e}", file=sys.stderr)
        return None


def save_cached_images(pdf_path: Path, images_by_page: Dict[int, List[Image.Image]]) -> None:
    """
    Save extracted images to cache.
    """
    cache_path = get_cache_path(pdf_path)

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(images_by_page, f)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}", file=sys.stderr)


def extract_images_from_page(page: fitz.Page) -> List[Image.Image]:
    """
    Extract all images from a PDF page.
    """
    images = []
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]

        try:
            # Extract image
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert to PIL Image
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
        except Exception as e:
            print(f"Warning: Could not extract image {img_index} from page: {e}", file=sys.stderr)
            continue

    return images


def get_extracted_images(pdf_path: Path) -> Dict[int, List[Image.Image]]:
    """
    Get extracted images from PDF, using cache if available.

    Returns:
        Dictionary mapping page numbers to lists of images
    """
    # Try to load cached images
    images_by_page = load_cached_images(pdf_path)
    if images_by_page:
        print("Using cached extracted images...")
        return images_by_page

    # Extract images if not cached
    print("Extracting images from PDF...")
    images_by_page = {}

    # Open PDF
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error: Could not open PDF {pdf_path}: {e}", file=sys.stderr)
        return {}

    # Extract images from each page
    doc_len = len(pdf_document)
    for page_num in range(doc_len):
        if (page_num + 1) % 50 == 0:
            print(f"  Extracting from page {page_num + 1}/{doc_len}...")
        page = pdf_document[page_num]
        page_images = extract_images_from_page(page)
        images_by_page[page_num] = page_images

    pdf_document.close()

    # Save to cache
    print("Saving extracted images to cache...")
    save_cached_images(pdf_path, images_by_page)

    return images_by_page


def find_image_in_pdf(pdf_path: Path, search_image_path: Path, images_by_page: Dict[int, List[Image.Image]], threshold: float) -> List[Tuple[int, float]]:
    """
    Find which pages in a PDF contain the given image using perceptual hashing.

    Args:
        pdf_path: Path to the PDF file
        search_image_path: Path to the image to search for
        images_by_page: Pre-extracted images from the PDF
        threshold: Similarity threshold (0-1) for matching

    Returns:
        List of tuples (page_number, similarity_score) where matches were found
    """
    # Load the search image
    try:
        search_image = Image.open(search_image_path)
    except Exception as e:
        print(f"Error: Could not load image {search_image_path}: {e}", file=sys.stderr)
        return []

    # Compare images using perceptual hashing
    print(f"Searching for matching images (threshold: {threshold:.2f})...")
    matches = []

    for page_num, page_images in images_by_page.items():
        # Compare each image with the search image
        for img in page_images:
            similarity = calculate_perceptual_similarity(img, search_image)

            if similarity >= threshold:
                matches.append((page_num + 1, similarity))  # +1 for human-readable page numbers
                break  # Found a match on this page, move to next page

    return matches


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Find which page(s) of a PDF contain a given image using perceptual hashing"
    )
    parser.add_argument("pdf", type=str, help="Path to the PDF file")
    parser.add_argument("image", type=str, help="Path to the image to search for")

    args = parser.parse_args()

    # Convert to Path objects
    pdf_path = Path(args.pdf)
    image_path = Path(args.image)

    # Validate inputs
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        return 1

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return 1

    # Extract images from PDF once (uses cache if available)
    images_by_page = get_extracted_images(pdf_path)
    if not images_by_page:
        print("Error: Could not extract images from PDF", file=sys.stderr)
        return 1

    # Auto-adjust threshold in parallel bunches of 16
    min_threshold = 0.50
    step = 0.05
    bunch_size = 16

    # Generate all thresholds
    all_thresholds = []
    threshold = 0.95
    while threshold >= min_threshold:
        all_thresholds.append(threshold)
        threshold -= step

    # Process in bunches
    best_matches = []
    best_threshold = None

    for i in range(0, len(all_thresholds), bunch_size):
        bunch = all_thresholds[i:i+bunch_size]
        print(f"Trying thresholds: {', '.join(f'{t:.2f}' for t in bunch)}")

        # Run searches in parallel for this bunch
        results = {}
        with ThreadPoolExecutor(max_workers=bunch_size) as executor:
            future_to_threshold = {
                executor.submit(find_image_in_pdf, pdf_path, image_path, images_by_page, t): t
                for t in bunch
            }

            for future in as_completed(future_to_threshold):
                t = future_to_threshold[future]
                matches = future.result()
                if len(matches) > 0:
                    results[t] = matches

        # If we found matches, pick the highest threshold
        if results:
            best_threshold = max(results.keys())
            best_matches = results[best_threshold]
            print(f"Found matches at threshold {best_threshold:.2f}, stopping search.")
            break

    # Sort matches by similarity (highest first)
    best_matches.sort(key=lambda x: x[1], reverse=True)

    # Display results (always verbose)
    if best_matches and best_threshold is not None:
        print(f"\nImage found on {len(best_matches)} page(s) at threshold {best_threshold:.2f}:")
        for page_num, similarity in best_matches:
            print(f"  Page {page_num} (similarity: {similarity:.2%})")
    else:
        print(f"\nImage not found in the PDF (tried thresholds down to {min_threshold:.2f}).")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
