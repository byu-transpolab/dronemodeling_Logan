import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bparser import BibTexParser

# Fields that IEEE BibTeX doesn't need or often breaks on
FIELDS_TO_REMOVE = [
    'urldate', 'month', 'address', 'language', 'organization',
    'doi', 'isbn', 'issn', 'url', 'howpublished', 'note'
]

def format_title(title):
    """Capitalize only first letter and protect acronyms."""
    if not title:
        return ""
    # Keep uppercase for acronyms (like UAV, AI)
    words = title.split()
    formatted = " ".join(
        [words[0].capitalize()] + [w if w.isupper() else w.lower() for w in words[1:]]
    )
    return formatted.strip()

def clean_entry(entry):
    """Clean up and standardize one bib entry."""
    for field in FIELDS_TO_REMOVE:
        entry.pop(field, None)

    # Standardize title capitalization
    if 'title' in entry:
        entry['title'] = format_title(entry['title'])

    # Clean authors (remove extra spaces)
    if 'author' in entry:
        entry['author'] = " and ".join(
            [a.strip() for a in entry['author'].replace("\n", " ").split(" and ")]
        )

    return entry

def convert_biblatex_to_ieee(input_bib, output_bib):
    """Convert a BibLaTeX .bib file to IEEE-compatible BibTeX."""
    parser = BibTexParser(common_strings=True)
    with open(input_bib, 'r', encoding='utf-8') as bibfile:
        bib_database = bibtexparser.load(bibfile, parser=parser)

    cleaned_entries = [clean_entry(entry) for entry in bib_database.entries]
    bib_database.entries = cleaned_entries

    writer = BibTexWriter()
    writer.indent = '  '
    writer.order_entries_by = ('year', 'author')

    with open(output_bib, 'w', encoding='utf-8') as bibfile:
        bibfile.write(writer.write(bib_database))

    print(f"✅ Converted '{input_bib}' → '{output_bib}' (IEEE compatible)")

if __name__ == "__main__":
    # Example usage:
    input_file = "ietc_paper/_extensions/jansim/bibliography_latex.bib"
    output_file = "ietc_paper/bibliography.bib"
    convert_biblatex_to_ieee(input_file, output_file)