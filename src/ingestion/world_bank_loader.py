"""World Bank GEP loader: section-aware text and table-row extraction.

Sections come from two levels of structure in the PDF:
  - chapter: the PDF bookmarks (Ch. 1, Box 1.1, regional outlooks, ...)
  - section: in-page subheadings, set in Helvetica at 10.5pt or larger
    (body text is Garamond), e.g. "Recent developments", "Outlook".

Every text block keeps the page it came from, so child chunks can cite
the exact page rather than the section's starting page.

Forecast tables are flattened by plain text extraction ("Euro area 0.5
1.0 1.4 0.8 ...") and lose their column headers, so they are parsed from
word positions into one self-describing record per row.
"""

import re
import unicodedata
from pathlib import Path

import pymupdf


RUNNING_HEADER = re.compile(
    r"^(GLOBAL ECONOMIC PROSPECTS \| JUNE 2026|CHAPTER \d+|\d{1,3}"
    r"|BOX \d+\.\d+.*\(continued\))$"
)
YEAR_COLUMN = re.compile(r"^20\d\d[ef]?$")
NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
TABLE_TITLE = re.compile(r"^TABLE \d+\.\d+")
BOX_TITLE = re.compile(r"^BOX \d+\.\d+")


def clean(text: str) -> str:
    # NFKC expands typographic ligatures such as "ﬁ" and "ﬂ".
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.replace(" -- ", " ").split())


def span_text(span: dict) -> str:
    """Rebuild span text from its glyphs, restoring the "Th" ligature.

    The report's Garamond font maps its "Th" ligature to a plain "T", so
    "The" is extracted as "Te". The ligature glyph is about 1.07x the font
    size wide versus 0.66x for a real "T", which tells them apart.
    """
    chars = []

    for char in span["chars"]:
        width = char["bbox"][2] - char["bbox"][0]

        if char["c"] == "T" and width > 0.9 * span["size"]:
            chars.append("Th")
        else:
            chars.append(char["c"])

    return "".join(chars)


def load_chapters(pdf) -> list[tuple[int, str]]:
    # Boxes are bookmarked where they start but not where they end, so the
    # chapter text after a box would be mislabelled; their titles are
    # picked up as in-page headings instead.
    toc = sorted(
        (page, clean(title))
        for _, title, page in pdf.get_toc()
        if page > 0 and not clean(title).startswith("Box ")
    )
    return toc or [(1, "World Bank report")]


def chapter_for_page(chapters: list[tuple[int, str]], page: int) -> str:
    title = chapters[0][1]

    for start, name in chapters:
        if start <= page:
            title = name

    return title


def is_body_font(span: dict) -> bool:
    return "Garamond" in span["font"] and span["size"] >= 8


def is_heading_line(spans: list[dict], text: str) -> bool:
    if not all("Helvetica" in s["font"] for s in spans):
        return False

    size = max(s["size"] for s in spans)

    # Box titles are 10pt bold ("BOX 1.1 How much will AI ...").
    return (
        (size >= 10.5 and 3 <= len(text) <= 90 and not TABLE_TITLE.match(text))
        or (size >= 9.5 and BOX_TITLE.match(text) is not None)
    )


def load_world_bank_sections(pdf_path: str) -> list[dict]:
    """Return sections, each with per-page text blocks.

    {"chapter", "title", "start_page", "end_page",
     "pages": [{"page": int, "text": str}, ...], "text": str}
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    sections = []
    current = None

    def start_section(chapter: str, title: str, page: int, size=None):
        nonlocal current

        if current and current["pages"]:
            sections.append(current)

        current = {
            "chapter": chapter,
            "title": title,
            "start_page": page,
            "end_page": page,
            "pages": [],
            "heading_size": size,
        }

    with pymupdf.open(path) as pdf:
        chapters = load_chapters(pdf)
        current_chapter = None

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            chapter = chapter_for_page(chapters, page_number)

            if chapter != current_chapter:
                current_chapter = chapter
                start_section(chapter, chapter, page_number)

            page_lines = []

            for block in page.get_text("rawdict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        span["text"] = span_text(span)

                    spans = [s for s in line["spans"] if s["text"].strip()]
                    text = clean("".join(s["text"] for s in spans))

                    if not spans or RUNNING_HEADER.match(text):
                        continue

                    if is_heading_line(spans, text):
                        size = round(max(s["size"] for s in spans))

                        if not page_lines and not current["pages"]:
                            # First heading under a chapter bookmark, or the
                            # second line of a wrapped heading.
                            if current["heading_size"] is None:
                                current["title"] = text
                                current["heading_size"] = size
                                continue

                            if current["heading_size"] == size:
                                current["title"] += f" {text}"
                                continue

                        if page_lines:
                            current["pages"].append({
                                "page": page_number,
                                "text": join_lines(page_lines),
                            })
                            current["end_page"] = page_number
                            page_lines = []

                        start_section(chapter, text, page_number, size)
                    elif all(is_body_font(s) for s in spans):
                        page_lines.append(text)

            if page_lines:
                current["pages"].append({
                    "page": page_number,
                    "text": join_lines(page_lines),
                })
                current["end_page"] = page_number

    if current and current["pages"]:
        sections.append(current)

    for section in sections:
        section.pop("heading_size")
        section["text"] = "\n".join(p["text"] for p in section["pages"])

    return sections


def join_lines(lines: list[str]) -> str:
    # Re-join words hyphenated across line breaks ("disrup-" + "tions").
    text = ""

    for line in lines:
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}" if text else line

    return text


def group_rows(words: list[tuple], tolerance: float = 2.5) -> list[list]:
    rows = []

    for word in sorted(words, key=lambda w: (round(w[3], 1), w[0])):
        if rows and abs(rows[-1][0][3] - word[3]) <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])

    return [sorted(row, key=lambda w: w[0]) for row in rows]


def parse_table(rows: list[list], title: str, subtitle: str) -> list[str]:
    header_index = next(
        (
            i for i, row in enumerate(rows)
            if sum(bool(YEAR_COLUMN.match(w[4])) for w in row) >= 3
        ),
        None,
    )

    if header_index is None:
        return []

    columns = []
    seen = set()

    for word in rows[header_index]:
        if YEAR_COLUMN.match(word[4]):
            # A repeated year starts the "change vs January" block.
            group = "revision" if word[4] in seen else "level"
            seen.add(word[4])
            columns.append({
                "label": word[4],
                "group": group,
                "x": (word[0] + word[2]) / 2,
                "x0": word[0],
            })

    first_x0 = min(c["x0"] for c in columns)
    records = []

    for row in rows[header_index + 1:]:
        label = " ".join(w[4] for w in row if w[2] < first_x0 - 2)
        # Drop footnote markers attached to the row label ("India 2").
        label = re.sub(r"(\s\d)+$", "", label).strip()

        values = {}

        for word in row:
            if word[2] < first_x0 - 2 or not NUMBER.match(word[4]):
                continue

            centre = (word[0] + word[2]) / 2
            column = min(columns, key=lambda c: abs(c["x"] - centre))
            values[(column["group"], column["label"])] = word[4]

        if not label or len(values) < 2:
            continue

        # Notes under the table ("3. Growth estimate for 2025 ...").
        if label.lower().startswith(("source", "note")) or re.match(r"^\d+\.\s", label):
            break

        levels = ", ".join(
            f"{c['label']}={values[('level', c['label'])]}"
            for c in columns
            if c["group"] == "level" and ("level", c["label"]) in values
        )
        revisions = ", ".join(
            f"{c['label']}={values[('revision', c['label'])]}"
            for c in columns
            if c["group"] == "revision" and ("revision", c["label"]) in values
        )

        # Row label first: with the long table title first, every row of a
        # table embeds almost identically and the country name is lost.
        record = f"{label} | {title}\n{label}: {levels}"

        if revisions:
            record += (
                f"\n{label} percentage-point change from January 2026 "
                f"projections: {revisions}"
            )

        records.append(f"{record}\n{subtitle}".strip())

    return records


def load_world_bank_tables(pdf_path: str) -> list[dict]:
    """Return one record per table row: {"page", "chapter", "title", "text"}."""
    records = []

    with pymupdf.open(pdf_path) as pdf:
        chapters = load_chapters(pdf)

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            rows = group_rows(page.get_text("words"))
            row_texts = [" ".join(w[4] for w in row) for row in rows]

            starts = [
                i for i, text in enumerate(row_texts)
                if TABLE_TITLE.match(text)
            ]

            for n, start in enumerate(starts):
                end = starts[n + 1] if n + 1 < len(starts) else len(rows)
                title = re.sub(r"\s\d$", "", row_texts[start])
                subtitle = next(
                    (
                        t for t in row_texts[start + 1:start + 4]
                        if t.startswith("(")
                    ),
                    "",
                )

                for text in parse_table(rows[start:end], title, subtitle):
                    records.append({
                        "page": page_number,
                        "chapter": chapter_for_page(chapters, page_number),
                        "title": title,
                        "text": text,
                    })

    return records


if __name__ == "__main__":
    path = "data/world_bank/GEP-Jun-2026.pdf"
    sections = load_world_bank_sections(path)
    tables = load_world_bank_tables(path)

    print(f"Sections detected: {len(sections)}")
    print(f"Table rows parsed: {len(tables)}")

    for section in sections[:25]:
        print(
            f"p{section['start_page']}-{section['end_page']}",
            f"{section['chapter'][:30]} > {section['title'][:50]}",
            f"({len(section['text'])} chars)",
        )

    for record in tables[:3]:
        print(f"\np{record['page']}: {record['text']}")
