#!/usr/bin/env bash
# convert_to_pdf.sh — Convert HTML documentation to PDF
# Usage: ./convert_to_pdf.sh <input.html> <output.pdf>

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <input.html> <output.pdf>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
  echo "Error: Input file '$INPUT' not found."
  exit 1
fi

# Try Chrome/Chromium first (headless print to PDF)
if command -v google-chrome &> /dev/null; then
  echo "Using Google Chrome to convert HTML to PDF..."
  google-chrome --headless --disable-gpu --print-to-pdf="$OUTPUT" "$INPUT"
  echo "PDF created: $OUTPUT"
elif command -v chromium &> /dev/null; then
  echo "Using Chromium to convert HTML to PDF..."
  chromium --headless --disable-gpu --print-to-pdf="$OUTPUT" "$INPUT"
  echo "PDF created: $OUTPUT"
elif command -v wkhtmltopdf &> /dev/null; then
  echo "Using wkhtmltopdf to convert HTML to PDF..."
  wkhtmltopdf "$INPUT" "$OUTPUT"
  echo "PDF created: $OUTPUT"
else
  echo "Error: Neither Chrome/Chromium nor wkhtmltopdf found."
  echo "Please install one of them to convert HTML to PDF."
  exit 1
fi
