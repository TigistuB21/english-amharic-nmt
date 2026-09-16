# Data Directory

This directory manages parallel corpus files for English-to-Amharic Neural Machine Translation.

## Directory Layout

- **`raw/`**: Contains original parallel datasets (e.g. parallel text files, TSV/CSV format) as downloaded from publicly available sources without modification.
- **`processed/`**: Contains cleaned, normalized, tokenized texts, vocabulary mappings, and serialized train/validation/test splits.

## Notes
- Large dataset files are tracked via `.gitignore` and should not be committed to Git.
- Preserve source attribution, licenses, and download URLs when adding raw data.
