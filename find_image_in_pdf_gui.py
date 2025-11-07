#!/usr/bin/env python3
"""
Web-based GUI for finding images in PDFs using perceptual hashing.
"""

import gradio as gr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the core functionality
from find_image_in_pdf import get_extracted_images, find_image_in_pdf


def search_image_in_pdf(pdf_file, image_file, progress=gr.Progress()):
    """
    Search for an image in a PDF file.

    Args:
        pdf_file: Uploaded PDF file
        image_file: Uploaded image file
        progress: Gradio progress tracker

    Returns:
        str: Formatted results
    """
    if pdf_file is None:
        return "❌ Please upload a PDF file"

    if image_file is None:
        return "❌ Please upload an image file"

    try:
        pdf_path = Path(pdf_file.name)
        image_path = Path(image_file.name)

        output = []
        output.append("🔍 Starting search...\n")
        output.append(f"📄 PDF: {pdf_path.absolute()}")
        output.append(f"🖼️  Image: {image_path.absolute()}\n")

        # Extract images from PDF
        progress(0.1, desc="Extracting images from PDF...")
        output.append("⏳ Extracting images from PDF...")

        images_by_page = get_extracted_images(pdf_path)

        if not images_by_page:
            return "\n".join(output) + "\n\n❌ Error: Could not extract images from PDF"

        output.append(f"✅ Extracted images from {len(images_by_page)} pages\n")

        # Auto-adjust threshold in parallel bunches
        progress(0.3, desc="Searching for matches...")
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
            output.append(f"🔎 Trying thresholds: {', '.join(f'{t:.2f}' for t in bunch)}")

            progress((0.3 + (i / len(all_thresholds)) * 0.6), desc=f"Testing thresholds {bunch[0]:.2f}-{bunch[-1]:.2f}...")

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
                output.append(f"✅ Found matches at threshold {best_threshold:.2f}!\n")
                break

        progress(0.9, desc="Formatting results...")

        # Sort matches by similarity (highest first)
        best_matches.sort(key=lambda x: x[1], reverse=True)

        # Display results
        output.append("=" * 70)
        if best_matches and best_threshold is not None:
            output.append(f"\n✅ Image found on {len(best_matches)} page(s) at threshold {best_threshold:.2f}\n")

            # Create formatted table
            output.append("┌────────────┬──────────────┐")
            output.append("│    Page    │  Similarity  │")
            output.append("├────────────┼──────────────┤")
            for page_num, similarity in best_matches:
                page_str = f"{page_num}".center(10)
                sim_str = f"{similarity:.2%}".center(12)
                output.append(f"│ {page_str} │ {sim_str} │")
            output.append("└────────────┴──────────────┘")
        else:
            output.append(f"\n❌ Image not found in the PDF (tried thresholds down to {min_threshold:.2f})")

        progress(1.0, desc="Complete!")
        return "\n".join(output)

    except Exception as e:
        import traceback
        error_output = [
            "❌ **Error occurred:**",
            "",
            str(e),
            "",
            "**Traceback:**",
            "```",
            traceback.format_exc(),
            "```"
        ]
        return "\n".join(error_output)


# Create Gradio interface
with gr.Blocks(
    title="Find Image in PDF",
    theme=gr.themes.Soft(),
    css=".monospace textarea { font-family: monospace !important; }"
) as app:
    gr.Markdown("# 🔍 Find Image in PDF")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                pdf_input = gr.File(
                    label="📄 Upload PDF File",
                    file_types=[".pdf"],
                    type="filepath",
                    scale=4
                )
                pdf_clear_btn = gr.Button("✖", scale=1, size="sm")

            with gr.Row():
                image_input = gr.File(
                    label="🖼️ Upload Image File",
                    file_types=["image"],
                    type="filepath",
                    scale=4
                )
                image_clear_btn = gr.Button("✖", scale=1, size="sm")

            search_btn = gr.Button("🔍 Search", variant="primary", size="lg")

        with gr.Column():
            output = gr.Textbox(
                label="Results",
                lines=25,
                max_lines=50,
                show_copy_button=True,
                elem_classes="monospace"
            )

    # Connect the search button
    search_btn.click(
        fn=search_image_in_pdf,
        inputs=[pdf_input, image_input],
        outputs=output
    )

    # Connect clear buttons
    pdf_clear_btn.click(
        fn=lambda: None,
        inputs=None,
        outputs=pdf_input
    )

    image_clear_btn.click(
        fn=lambda: None,
        inputs=None,
        outputs=image_input
    )


def main():
    """Launch the web GUI."""
    print("🚀 Starting Find Image in PDF web interface...")
    print("📱 Open the URL below in your browser")
    app.launch(share=False)


if __name__ == "__main__":
    main()
