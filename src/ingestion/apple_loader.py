"""Apple Product Environmental Report loader: text, tables, chart, endnotes.

Produces small, self-describing records so each one embeds well:
  - text:     paragraph chunks prefixed with the page title; every footnote
              marker in the chunk has its endnote text appended
  - footnote: one record per endnote ("Endnote 16: ...")
  - table:    one record per row with its column headers
              ("Sleep: 100V=0.56W, 115V=0.57W, 230V=0.59W")
  - chart:    the bar chart's labels grouped per bar, top to bottom

Each record keeps "content" (embedded) and "text" (content plus linked
endnotes, passed to the LLM).

Footnote markers are superscripts: digit spans set noticeably smaller than
the rest of their line. Endnotes start with such a marker on the
"Endnotes" pages.
"""

import json
import re
import unicodedata
from pathlib import Path

import pymupdf


PAGE_FOOTER = re.compile(r"^(iMac\s*\|\s*Product Environmental Report|\d{1,2})$")
SUPERSCRIPT_RATIO = 0.75
CHUNK_CHARS = 700


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def is_marker(span: dict, line_size: float) -> bool:
    return (
        span["size"] < SUPERSCRIPT_RATIO * line_size
        and re.fullmatch(r"\d{1,2}", span["text"].strip()) is not None
    )


def page_lines(page, skip_rects: list) -> list[dict]:
    """Lines with their text, footnote markers and position."""
    lines = []

    for block_index, block in enumerate(page.get_text("dict")["blocks"]):
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]

            if not spans:
                continue

            box = pymupdf.Rect(line["bbox"])

            if any(box.intersects(rect) for rect in skip_rects):
                continue

            line_size = max(s["size"] for s in spans)
            parts = []
            markers = []
            previous = None

            for span in spans:
                if is_marker(span, line_size):
                    markers.append(int(span["text"].strip()))
                    parts.append(f"[{span['text'].strip()}]")
                else:
                    # Adjacent spans can drop the space between them
                    # ("established" + "iMac"); restore it from the gap.
                    if (
                        previous
                        and span["bbox"][0] - previous["bbox"][2] > 0.5
                        and not previous["text"].endswith(" ")
                        and not span["text"].startswith(" ")
                    ):
                        parts.append(" ")

                    parts.append(span["text"])

                previous = span

            text = clean("".join(parts))

            if PAGE_FOOTER.match(text):
                continue

            lines.append({
                "text": text,
                "markers": markers,
                "starts_with_marker": is_marker(spans[0], line_size),
                "size": line_size,
                "block": block_index,
                "bbox": box,
            })

    return lines


def page_title(lines: list[dict]) -> str:
    if not lines:
        return ""

    size = max(line["size"] for line in lines)
    return " ".join(line["text"] for line in lines if line["size"] == size)


# ---------------------------------------------------------------- tables

def group_word_rows(words: list, tolerance: float = 5) -> list[list]:
    """Group words into visual rows; chained so wrapped cells stay together."""
    rows = []

    for word in sorted(words, key=lambda w: (w[3], w[0])):
        if rows and word[3] - rows[-1][-1][3] <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])

    return [sorted(row, key=lambda w: w[0]) for row in rows]


def column_starts(table) -> list[float]:
    starts = []

    for row in table.rows:
        for cell in row.cells:
            if cell and all(abs(cell[0] - x) > 3 for x in starts):
                starts.append(cell[0])

    return sorted(starts) if len(starts) >= 2 else []


def is_chart_like(table) -> bool:
    cells = [cell for row in table.rows for cell in row.cells]
    return sum(cell is None for cell in cells) > len(cells) / 2


def assign_cells(row: list, starts: list[float]) -> dict[int, str]:
    cells = {}

    for word in sorted(row, key=lambda w: (round(w[3]), w[0])):
        column = max(
            (i for i, x in enumerate(starts) if x <= word[0] + 3),
            default=0,
        )
        cells.setdefault(column, []).append(word[4])

    return {i: clean(" ".join(ws)) for i, ws in cells.items()}


def strip_marker(label: str) -> tuple[str, list[int]]:
    # Table labels carry footnote markers glued on ("achieved7").
    match = re.match(r"^(.*?[A-Za-z)])(\d{1,2})$", label)

    if match:
        return match.group(1), [int(match.group(2))]

    return label, []


def extract_tables(
    page,
    page_number: int,
    page_heading: str,
) -> tuple[list[dict], list]:
    """Return table-row records and the regions they cover.

    Each row is embedded on its own ("content") so the right row is found,
    but the LLM gets the whole table ("text"): these tables are small, and
    a lone row hides its siblings (Production 50% without Product use 45%).

    PyMuPDF only detects the shaded header band (and some shaded rows)
    of these borderless tables, so the header band gives the columns and
    the rows below it are rebuilt from word positions.
    """
    detected = [
        t for t in page.find_tables().tables
        if not is_chart_like(t)
    ]
    detected.sort(key=lambda t: t.bbox[1])

    records = []
    regions = []
    consumed_until = -1

    for table in detected:
        x0, y0, x1, y1 = table.bbox

        if y0 < consumed_until:
            continue

        starts = column_starts(table)

        if not starts:
            continue

        words = [
            w for w in page.get_text("words")
            if w[1] >= y0 - 2 and x0 - 5 <= w[0] <= x1 + 5
        ]
        header, body = [], []
        previous_bottom = None

        for row in group_word_rows(words):
            cells = assign_cells(row, starts)
            in_header = row[0][3] <= y1 + 1
            bottom = max(w[3] for w in row)

            # Rows sit ~15-17pt apart; a larger gap means the table ended
            # and a note or the next paragraph begins.
            if (
                not in_header
                and previous_bottom is not None
                and min(w[3] for w in row) - previous_bottom > 19
            ):
                break

            previous_bottom = bottom

            (header if in_header else body).append((row, cells))

        if not body:
            continue

        # In a multi-row header, a row of tightly spaced words is a title
        # spanning several columns ("Power consumption for iMac").
        def is_spanning(row):
            gaps = [b[0] - a[2] for a, b in zip(row, row[1:])]
            return len(header) > 1 and max(gaps, default=0) < 15

        caption = " ".join(
            clean(" ".join(w[4] for w in row))
            for row, _ in header if is_spanning(row)
        )
        names = {}

        for row, cells in header:
            if not is_spanning(row):
                for i, text in cells.items():
                    names[i] = f"{names.get(i, '')} {text}".strip()

        parent = ""
        title = " - ".join(
            t for t in (page_heading, caption, names.get(0)) if t
        ) or "Table"
        rows = []

        for row, cells in body:
            label, notes = strip_marker(cells.get(0, ""))

            if label.startswith("•"):
                label = f"{parent} > {label.lstrip('• ')}"
            else:
                parent = label

            values = ", ".join(
                f"{names.get(i, f'col{i}')} = {cells[i]}"
                for i in sorted(cells) if i > 0
            )

            if not values:
                continue

            rows.append((f"{label}: {values}", notes))

        table_text = f"Table: {title}\n" + "\n".join(r for r, _ in rows)
        table_markers = [n for _, notes in rows for n in notes]

        for row_text, _ in rows:
            records.append({
                "type": "table",
                "page": page_number,
                "content": f"Table: {title}\n{row_text}",
                "text": table_text,
                "markers": table_markers,
            })

        bottom = body[-1][0][-1][3]
        consumed_until = bottom
        regions.append(pymupdf.Rect(x0, y0, x1, bottom + 2))

    return records, regions


# ----------------------------------------------------------------- chart

def extract_chart(page, page_number: int, title: str) -> tuple[list, list]:
    """Describe a stacked bar chart from the labels inside its drawing area.

    Bar values and segment names are in the text layer; segment sizes are
    only drawn, so they are not recoverable without vision.
    """
    # Stacked bars: >= 2 columns of >= 3 filled segments whose heights
    # vary a lot. Table row shading has equal heights, so it is excluded.
    columns = {}

    for drawing in page.get_drawings():
        rect = drawing["rect"]

        if drawing.get("fill") and 40 < rect.width < 150 and rect.height > 8:
            columns.setdefault(round(rect.x0), []).append(rect)

    bars = [
        rects for rects in columns.values()
        if len(rects) >= 3
        and max(r.height for r in rects) > 2 * min(r.height for r in rects)
    ]

    if len(bars) < 2:
        return [], []

    region = pymupdf.Rect()

    for rects in bars:
        for rect in rects:
            region |= rect

    bars_bottom = region.y1

    # Bar totals sit above the bars and footnoted labels just below.
    region = pymupdf.Rect(region.x0, region.y0 - 40, region.x1, region.y1 + 45)

    spans = []

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            box = pymupdf.Rect(line["bbox"])

            if not box.intersects(region):
                continue

            line_size = max(s["size"] for s in line["spans"])
            text = clean("".join(
                f"[{s['text'].strip()}]" if is_marker(s, line_size)
                else s["text"]
                for s in line["spans"]
            ))

            if text and not PAGE_FOOTER.match(text):
                spans.append((box, text))

    # Labels below the bars are call-outs, not bars of their own.
    callouts = [text for box, text in spans if box.y0 > bars_bottom]
    spans = [(box, text) for box, text in spans if box.y0 <= bars_bottom]

    # Cluster labels into bars by horizontal centre.
    bars = []

    for box, text in sorted(spans, key=lambda s: (s[0].x0 + s[0].x1) / 2):
        centre = (box.x0 + box.x1) / 2

        if bars and centre - bars[-1]["centre"] < 40:
            bars[-1]["items"].append((box, text))
        else:
            bars.append({"centre": centre, "items": [(box, text)]})

    lines = [f"Chart: {title} (stacked bar chart)"]
    totals = []

    for number, bar in enumerate(bars, start=1):
        items = sorted(bar["items"], key=lambda i: i[0].y0)
        merged = []

        for box, text in items:
            # Re-join labels wrapped over several lines.
            if merged and box.y0 - merged[-1][0].y1 < 4:
                merged[-1] = (merged[-1][0] | box, f"{merged[-1][1]} {text}")
            else:
                merged.append((box, text))

        labels = [text for _, text in merged]
        totals.append(re.sub(r"\[\d+\]", "", labels[0]))
        lines.append(
            f"Bar {number}: {labels[0]} - segments top to bottom: "
            + " | ".join(labels[1:])
        )

    if callouts:
        lines.append("Call-out below the bars: " + " ".join(callouts))

    # Lead with the bar totals: they carry the chart's numbers, and the
    # segment lists would otherwise dominate the embedding.
    lines.insert(1, "Bar totals, left to right: " + " -> ".join(totals))

    markers = [int(m) for m in re.findall(r"\[(\d+)\]", "\n".join(lines))]
    record = {
        "type": "chart",
        "page": page_number,
        "text": "\n".join(lines),
        "markers": markers,
    }

    return [record], [region]


# ------------------------------------------------------------- endnotes

def extract_endnotes(lines: list[dict]) -> dict[int, str]:
    notes = {}
    current = None

    for line in lines:
        if line["starts_with_marker"]:
            match = re.match(r"^\[(\d+)\]\s*(.*)$", line["text"])

            if match:
                current = int(match.group(1))
                notes[current] = match.group(2)
                continue

        if current is not None and not line["text"].startswith("©"):
            notes[current] += f" {line['text']}"

    return {n: clean(text) for n, text in notes.items()}


# ------------------------------------------------------------------ text

def chunk_page(lines: list[dict]) -> list[dict]:
    chunks = []
    current = {"text": "", "markers": [], "block": None}

    for line in lines:
        new_block = line["block"] != current["block"]

        if current["text"] and new_block and len(current["text"]) > CHUNK_CHARS:
            chunks.append(current)
            current = {"text": "", "markers": [], "block": None}

        joiner = "\n" if new_block else " "
        current["text"] = f"{current['text']}{joiner}{line['text']}".strip()
        current["markers"] += line["markers"]
        current["block"] = line["block"]

    if current["text"]:
        chunks.append(current)

    return chunks


def with_notes(text: str, markers: list[int], notes: dict[int, str]) -> str:
    linked = [
        f"[{n}] {notes[n]}"
        for n in dict.fromkeys(markers) if n in notes
    ]

    if not linked:
        return text

    return text + "\nLinked endnotes:\n" + "\n".join(linked)


def extract_apple_content(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    records = []
    text_records = []
    notes = {}

    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            title = page_title(page_lines(page, []))

            tables, table_regions = extract_tables(page, page_number, title)
            charts, chart_regions = extract_chart(page, page_number, title)
            lines = page_lines(page, table_regions + chart_regions)

            if title == "Endnotes" or "Endnotes" in title:
                notes.update(extract_endnotes(lines))
            else:
                for chunk in chunk_page(lines):
                    text_records.append({
                        "type": "text",
                        "page": page_number,
                        "text": f"[{title}]\n{chunk['text']}",
                        "markers": chunk["markers"],
                    })

            records += tables + charts

        note_pages = {}

        for page_index, page in enumerate(pdf):
            for line in page_lines(page, []):
                if line["starts_with_marker"] and line["markers"]:
                    note_pages.setdefault(line["markers"][0], page_index + 1)

    records = text_records + records

    # "content" is what gets embedded; "text" adds the linked endnotes for
    # the LLM. Long endnotes would otherwise dilute the record's embedding.
    for record in records:
        record.setdefault("content", record["text"])
        record["text"] = with_notes(record["text"], record["markers"], notes)

    for number, text in sorted(notes.items()):
        records.append({
            "type": "footnote",
            "page": note_pages.get(number, 11),
            "text": f"Endnote {number}: {text}",
            "content": f"Endnote {number}: {text}",
            "markers": [number],
        })

    for index, record in enumerate(records):
        record["id"] = f"{record['type']}-p{record['page']}-{index}"

    return records


if __name__ == "__main__":
    records = extract_apple_content("data/apple_imac/apple_imac_report.pdf")

    counts = {}

    for record in records:
        counts[record["type"]] = counts.get(record["type"], 0) + 1

    print(f"Extracted records: {len(records)} {counts}")

    with open(
        "data/apple_imac/extracted_records.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
