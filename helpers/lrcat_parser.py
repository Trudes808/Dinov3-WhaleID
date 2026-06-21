"""Read expert labels from a Lightroom catalog.

The Lightroom catalog format is a SQLite database, but table details vary across
Lightroom versions. These helpers keep the assumptions narrow and expose schema
inspection utilities so catalog parsing can be debugged directly from the
notebook.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any


DATE_RE = re.compile(r"(?P<year>20\d{2})[-_](?P<month>\d{1,2})[-_](?P<day>\d{1,2})")
ANIMAL_RE = re.compile(r"G(?P<group>\d{1,2})A(?P<animal>\d{1,2})", re.IGNORECASE)
SAME_AS_RE = re.compile(r"same\s+as\s+G(?P<group>\d{1,2})A(?P<animal>\d{1,2})", re.IGNORECASE)


def normalize_date(value: str | None) -> str | None:
    """Return YYYY-MM-DD from flexible Lightroom/folder date text."""
    if not value:
        return None
    match = DATE_RE.search(str(value))
    if not match:
        return None
    try:
        normalized = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None
    return normalized.isoformat()


def normalize_animal_code(group_value: str | int, animal_value: str | int) -> str:
    """Normalize a Lightroom animal code to G##A##."""
    return f"G{int(group_value):02d}A{int(animal_value):02d}"


def parse_animal_collection_name(collection_name: str, hierarchy: str | None = None) -> dict[str, Any] | None:
    """Extract date and G##A## labels from an expert animal collection name."""
    animal_match = ANIMAL_RE.search(collection_name or "")
    if not animal_match:
        return None

    date_label = normalize_date(collection_name) or normalize_date(hierarchy)
    animal_code = normalize_animal_code(animal_match.group("group"), animal_match.group("animal"))
    same_as_match = SAME_AS_RE.search(collection_name or "")
    same_as_code = None
    same_as_whale_id = None
    if same_as_match:
        same_as_code = normalize_animal_code(same_as_match.group("group"), same_as_match.group("animal"))
        same_as_whale_id = f"{date_label}-{same_as_code}" if date_label else None

    role_notes = collection_name[animal_match.end() :].strip(" _-\t")
    return {
        "day_label": date_label,
        "ground_truth_group": int(animal_match.group("group")),
        "ground_truth_animal": int(animal_match.group("animal")),
        "ground_truth_code": animal_code,
        "ground_truth_whale_id": f"{date_label}-{animal_code}" if date_label else animal_code,
        "same_as_code": same_as_code,
        "same_as_whale_id": same_as_whale_id,
        "role_notes": role_notes,
    }


def _connect_readonly(lrcat_path: str | Path) -> sqlite3.Connection:
    catalog_path = Path(lrcat_path).expanduser().resolve()
    uri = f"file:{catalog_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def inspect_schema(lrcat_path: str | Path) -> list[dict[str, Any]]:
    """Return table names, row counts, and columns for notebook inspection."""
    with _connect_readonly(lrcat_path) as connection:
        cursor = connection.cursor()
        table_names = [
            row["name"]
            for row in cursor.execute("select name from sqlite_master where type='table' order by name")
        ]
        schema_rows: list[dict[str, Any]] = []
        for table_name in table_names:
            columns = [row["name"] for row in cursor.execute(f'pragma table_info("{table_name}")')]
            row_count = cursor.execute(f'select count(*) as row_count from "{table_name}"').fetchone()[
                "row_count"
            ]
            schema_rows.append({"table_name": table_name, "row_count": row_count, "columns": columns})
    return schema_rows


def _collection_hierarchy(cursor: sqlite3.Cursor) -> dict[int, str]:
    rows = {
        int(row["id_local"]): {"name": row["name"], "parent": row["parent"]}
        for row in cursor.execute("select id_local, name, parent from AgLibraryCollection")
    }
    hierarchy_by_id: dict[int, str] = {}
    for collection_id in rows:
        names: list[str] = []
        current_id: int | None = collection_id
        seen_ids: set[int] = set()
        while current_id in rows and current_id not in seen_ids:
            seen_ids.add(current_id)
            names.append(str(rows[current_id]["name"]))
            parent_id = rows[current_id]["parent"]
            current_id = int(parent_id) if parent_id is not None else None
        hierarchy_by_id[collection_id] = " / ".join(reversed(names))
    return hierarchy_by_id


def _catalog_image_path(
    image_root: str | Path | None,
    path_from_root: str | None,
    base_name: str | None,
    extension: str | None,
) -> str | None:
    if not path_from_root or not base_name:
        return None
    filename = f"{base_name}.{extension}" if extension else str(base_name)
    relative_path = PurePosixPath(path_from_root) / filename
    if image_root:
        return str(Path(image_root).expanduser() / Path(*relative_path.parts))
    return str(relative_path)


def parse_lrcat_ground_truth(
    lrcat_path: str | Path,
    image_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return one record per image-to-animal-collection membership."""
    query = """
        select
            ci.collection as collection_id,
            ci.positionInCollection as position_in_collection,
            c.name as collection_name,
            c.parent as collection_parent,
            i.id_local as lightroom_image_id,
            i.captureTime as capture_time,
            i.originalCaptureTime as original_capture_time,
            i.colorLabels as color_label,
            i.rating as rating,
            i.pick as pick,
            i.fileWidth as file_width,
            i.fileHeight as file_height,
            f.id_local as file_id,
            f.baseName as base_name,
            f.extension as extension,
            f.originalFilename as original_filename,
            fo.pathFromRoot as path_from_root,
            root.absolutePath as catalog_root_absolute_path
        from AgLibraryCollectionImage ci
        join AgLibraryCollection c on c.id_local = ci.collection
        join Adobe_images i on i.id_local = ci.image
        join AgLibraryFile f on f.id_local = i.rootFile
        join AgLibraryFolder fo on fo.id_local = f.folder
        join AgLibraryRootFolder root on root.id_local = fo.rootFolder
        order by c.name, ci.positionInCollection, f.baseName
    """
    records: list[dict[str, Any]] = []
    with _connect_readonly(lrcat_path) as connection:
        cursor = connection.cursor()
        hierarchy_by_id = _collection_hierarchy(cursor)
        for row in cursor.execute(query):
            collection_id = int(row["collection_id"])
            hierarchy = hierarchy_by_id.get(collection_id, row["collection_name"])
            parsed_label = parse_animal_collection_name(row["collection_name"], hierarchy)
            if not parsed_label:
                continue
            image_path = _catalog_image_path(
                image_root=image_root,
                path_from_root=row["path_from_root"],
                base_name=row["base_name"],
                extension=row["extension"],
            )
            filename = Path(image_path).name if image_path else None
            color_label = (row["color_label"] or "").strip()
            color_label_lower = color_label.lower()
            capture_date = normalize_date(row["capture_time"]) or parsed_label["day_label"]
            records.append(
                {
                    "image_path": image_path,
                    "catalog_relative_path": str(PurePosixPath(row["path_from_root"] or "") / filename)
                    if filename
                    else None,
                    "filename": filename,
                    "image_stem": row["base_name"],
                    "extension": (row["extension"] or "").lower(),
                    "capture_time": row["capture_time"],
                    "original_capture_time": row["original_capture_time"],
                    "capture_date": capture_date,
                    "day_label": parsed_label["day_label"] or capture_date,
                    "lightroom_image_id": row["lightroom_image_id"],
                    "lightroom_file_id": row["file_id"],
                    "collection_id": collection_id,
                    "collection_name": row["collection_name"],
                    "collection_hierarchy": hierarchy,
                    "position_in_collection": row["position_in_collection"],
                    "catalog_root_absolute_path": row["catalog_root_absolute_path"],
                    "color_label": color_label,
                    "is_scarring_study_green": color_label_lower == "green",
                    "is_other_study_yellow": color_label_lower == "yellow",
                    "rating": row["rating"],
                    "pick": row["pick"],
                    "file_width": row["file_width"],
                    "file_height": row["file_height"],
                    **parsed_label,
                }
            )
    return records


def collapse_ground_truth_by_image(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse collection-membership records to one ground-truth row per image."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("image_path") or record.get("lightroom_image_id"))
        grouped[key].append(record)

    collapsed_records: list[dict[str, Any]] = []
    for image_key, image_records in grouped.items():
        first_record = image_records[0]
        whale_ids = sorted({str(record["ground_truth_whale_id"]) for record in image_records})
        collection_names = sorted({str(record["collection_name"]) for record in image_records})
        same_as_whale_ids = sorted(
            {str(record["same_as_whale_id"]) for record in image_records if record.get("same_as_whale_id")}
        )
        collapsed_records.append(
            {
                "image_key": image_key,
                "image_path": first_record.get("image_path"),
                "catalog_relative_path": first_record.get("catalog_relative_path"),
                "filename": first_record.get("filename"),
                "image_stem": first_record.get("image_stem"),
                "extension": first_record.get("extension"),
                "capture_time": first_record.get("capture_time"),
                "capture_date": first_record.get("capture_date"),
                "day_label": first_record.get("day_label"),
                "lightroom_image_id": first_record.get("lightroom_image_id"),
                "ground_truth_whale_id": whale_ids[0] if whale_ids else None,
                "ground_truth_whale_ids": "|".join(whale_ids),
                "ground_truth_label_count": len(whale_ids),
                "has_multiple_ground_truth_labels": len(whale_ids) > 1,
                "same_as_whale_ids": "|".join(same_as_whale_ids),
                "all_collection_names": "|".join(collection_names),
                "color_label": first_record.get("color_label"),
                "is_scarring_study_green": any(bool(record.get("is_scarring_study_green")) for record in image_records),
                "is_other_study_yellow": any(bool(record.get("is_other_study_yellow")) for record in image_records),
                "rating": first_record.get("rating"),
                "pick": first_record.get("pick"),
                "file_width": first_record.get("file_width"),
                "file_height": first_record.get("file_height"),
            }
        )
    return sorted(collapsed_records, key=lambda record: str(record.get("image_path") or ""))


def summarize_ground_truth(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact counts for print/debug cells."""
    day_counts = Counter(record.get("day_label") for record in records if record.get("day_label"))
    color_counts = Counter(record.get("color_label") or "unlabeled" for record in records)
    whale_ids = {record.get("ground_truth_whale_id") for record in records if record.get("ground_truth_whale_id")}
    return {
        "records": len(records),
        "unique_whales": len(whale_ids),
        "days": len(day_counts),
        "green_scarring_images": sum(bool(record.get("is_scarring_study_green")) for record in records),
        "yellow_study_images": sum(bool(record.get("is_other_study_yellow")) for record in records),
        "multiple_label_images": sum(bool(record.get("has_multiple_ground_truth_labels")) for record in records),
        "top_days": day_counts.most_common(12),
        "color_labels": color_counts.most_common(),
    }


def write_records_csv(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write dictionaries to CSV, converting nested values to JSON strings."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {}
            for key in fieldnames:
                value = record.get(key)
                if isinstance(value, (list, tuple, dict)):
                    row[key] = json.dumps(value, sort_keys=True)
                else:
                    row[key] = value
            writer.writerow(row)
    return path
