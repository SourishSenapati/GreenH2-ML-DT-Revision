"""
Script to verify DOIs in a markdown file.
"""
import re
import requests


def verify_references(file_path):
    """
    Extracts DOIs from the file and checks their validity using requests.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find DOIs - adjusting to catch the format in the file
    # Matches "DOI: 10.xxxx/..." or "DOI: [10.xxxx/...](url)"
    doi_pattern = r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+'

    dois = re.findall(doi_pattern, content)

    # Clean up DOIs (remove markdown link syntax if caught)
    clean_dois = []
    for d in dois:
        # If it grabbed trailing chars
        d = d.split(']')[0]
        clean_dois.append(d)

    clean_dois = list(set(clean_dois))

    print(f"Found {len(clean_dois)} unique DOIs.")

    valid_count = 0

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }

    for doi in clean_dois:
        url = f"https://doi.org/{doi}"
        try:
            # Follow redirects to see if it resolves
            r = requests.get(url, headers=headers,
                             allow_redirects=True, timeout=10)
            if r.status_code < 400:
                print(f"[OK] {doi}")
                valid_count += 1
            elif r.status_code == 403:
                # Some publishers might still block even with headers, but it's worth a try.
                # If verified manually, we can whitelist or print a specific warning.
                # For now, let's treat as fail but with a specific message.
                print(
                    f"[WARN] {doi} (Status: 403 - "
                    "Likely Bot Protection. Manual verify recommended.)")
                # If we know it's valid (Zhao et al), we could count it, but safer to warn.
            else:
                print(f"[FAIL] {doi} (Status: {r.status_code})")
        except requests.RequestException as e:
            print(f"[ERROR] {doi}: {e}")

    print(f"\nSummary: {valid_count}/{len(clean_dois)} valid.")

    if valid_count == len(clean_dois) and len(clean_dois) > 0:
        print("ALL DOIs VALID.")
    else:
        print("SOME DOIs FAILED.")


if __name__ == "__main__":
    verify_references(
        'd:/PROJECT/SCI PAPERS/GreenH2-ML-DT-Revision/drafts/lit_v2.md')
