"""Stage 1: profile the raw GTFS files -> reports/schema_profile.json.

For every file and column: row count, inferred type, null share, distinct count, examples, and whether the
column is part of the GTFS Schedule reference (columns outside it are flagged as non-standard)."""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW, REPORTS, write_json  # noqa: E402

# GTFS Schedule reference fields for the files this feed uses (gtfs.org/documentation/schedule/reference)
GTFS_REF = {
    "agency.txt": "agency_id agency_name agency_url agency_timezone agency_lang agency_phone agency_fare_url agency_email",
    "feed_info.txt": "feed_publisher_name feed_publisher_url feed_lang default_lang feed_start_date feed_end_date feed_version feed_contact_email feed_contact_url",
    "routes.txt": "route_id agency_id route_short_name route_long_name route_desc route_type route_url route_color route_text_color route_sort_order continuous_pickup continuous_drop_off network_id",
    "stops.txt": "stop_id stop_code stop_name tts_stop_name stop_desc stop_lat stop_lon zone_id stop_url location_type parent_station stop_timezone wheelchair_boarding level_id platform_code",
    "trips.txt": "route_id service_id trip_id trip_headsign trip_short_name direction_id block_id shape_id wheelchair_accessible bikes_allowed",
    "stop_times.txt": "trip_id arrival_time departure_time stop_id location_group_id location_id stop_sequence stop_headsign start_pickup_drop_off_window end_pickup_drop_off_window pickup_type drop_off_type continuous_pickup continuous_drop_off shape_dist_traveled timepoint pickup_booking_rule_id drop_off_booking_rule_id",
    "calendar_dates.txt": "service_id date exception_type",
    "transfers.txt": "from_stop_id to_stop_id from_route_id to_route_id from_trip_id to_trip_id transfer_type min_transfer_time",
}
TYPES = [("integer", r"-?\d+"), ("decimal", r"-?\d+\.\d+"), ("gtfs_time", r"\d{1,2}:\d{2}:\d{2}"),
         ("gtfs_date", r"\d{8}"), ("url", r"https?://\S+"), ("hex_color", r"[0-9A-Fa-f]{6}")]


def infer(vals):
    vals = [v for v in vals if v != ""]
    if not vals:
        return "empty"
    for name, rx in TYPES:
        if all(re.fullmatch(rx, v) for v in vals):
            return name
    return "string"


def main():
    prof = {}
    for f in sorted(RAW.glob("*.txt")):
        rows = list(csv.DictReader(open(f, newline="", encoding="utf-8-sig")))
        ref = set(GTFS_REF.get(f.name, "").split())
        cols = {}
        for c in (rows[0].keys() if rows else []):
            vals = [r[c] for r in rows]
            nn = [v for v in vals if v != ""]
            cols[c] = {"type": infer(vals), "null_share": round(1 - len(nn) / len(vals), 3) if vals else 1.0,
                       "distinct": len(set(nn)), "examples": sorted(set(nn))[:3],
                       "gtfs_standard": c in ref if ref else None}
            if cols[c]["type"] == "gtfs_time":
                cols[c]["max_hour"] = max(int(v.split(":")[0]) for v in nn)
        prof[f.name] = {"rows": len(rows), "columns": cols}
    write_json(REPORTS / "schema_profile.json", prof)
    nonstd = [f"{f}:{c}" for f, p in prof.items() for c, d in p["columns"].items() if d["gtfs_standard"] is False]
    print(f"profile_schema: {len(prof)} files, {sum(p['rows'] for p in prof.values())} rows; "
          f"non-standard columns: {', '.join(nonstd) or 'none'} -> reports/schema_profile.json")


if __name__ == "__main__":
    main()
