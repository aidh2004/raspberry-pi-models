import fitz
import sys

for pdf_path in sys.argv[1:]:
    print(f"\n{'='*80}")
    print(f"FILE: {pdf_path}")
    print(f"{'='*80}")
    doc = fitz.open(pdf_path)
    for page in doc:
        print(page.get_text())
    doc.close()
