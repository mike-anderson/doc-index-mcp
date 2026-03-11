"""Download test fixture PDFs for integration tests."""

import os
import urllib.request

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

FIXTURES = {
    "nist-csf-2.0.pdf": "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
    "nist-800-53r5.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
    "economic-report-2024.pdf": "https://www.govinfo.gov/content/pkg/ERP-2024/pdf/ERP-2024.pdf",
}


def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    for filename, url in FIXTURES.items():
        path = os.path.join(FIXTURES_DIR, filename)
        if os.path.exists(path):
            print(f"  Already exists: {filename}")
            continue
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Saved {filename} ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
