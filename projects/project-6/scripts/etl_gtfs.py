"""Stage 3: GTFS -> CCO instance graph (data/instances.ttl), implementing patterns/gtfs_patterns.json.

Normalizations: GTFS times may exceed 24:00:00 (service continuing after midnight); they are kept raw as text and
also normalized to xsd:dateTime = service date 00:00 local + offset, in America/New_York. peak_offpeak (a
non-standard LIRR column) becomes the boolean isPeakService. Raw values that fail validation (e.g. an invalid
time zone name) are loaded unchanged so QC can report them, never silently fixed."""
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

sys.path.insert(0, str(Path(__file__).parent))
from common import CCO, INSTANCES, OBO, P6, RAW, load_spec  # noqa: E402

C, B, X = Namespace(CCO), Namespace(OBO), Namespace(P6)
ID = Namespace(P6 + "id/")
T = {  # CCO / BFO terms
    "RailStation": X.RailStation, "Record": X.GTFSRecord, "CoordRec": X.StationCoordinateRecord,
    "Route": X.TransitRouteSpecification, "Trip": X.TrainTripSpecification, "Stop": X.ScheduledStopSpecification,
    "GeoLoc": C.ont00000487, "Name": C.ont00000003, "Code": C.ont00000077, "ArbId": C.ont00000923,
    "TZ": C.ont00000829, "TOD": C.ont00000827, "Org": C.ont00001180,
    "designates": C.ont00001916, "is_about": C.ont00001808, "uses_tz": C.ont00001908, "text": C.ont00001765,
    "lat": C.ont00001766, "lon": C.ont00001764, "datetime": C.ont00001767,
    "located_in": B.BFO_0000171, "carrier_of": B.BFO_0000101, "part_of": B.BFO_0000176,
}


def rows(name):
    return list(csv.DictReader(open(RAW / name, newline="", encoding="utf-8-sig")))


class Builder:
    def __init__(self):
        self.g = Graph()
        for p, ns in (("p6", X), ("id", ID), ("cco", C), ("obo", B)):
            self.g.bind(p, ns)

    def add(self, s, p, o):
        self.g.add((s, p, o))

    def record(self, iri, carries=None, label=None):
        self.add(iri, RDF.type, T["Record"])
        if carries is not None:
            self.add(iri, T["carrier_of"], carries)
        if label:
            self.add(iri, RDFS.label, Literal(label))
        return iri

    def designation(self, kind, owner, key, text):
        """Designative ICE (Name/Code/ArbId) designating owner, carried by a record holding the text."""
        d = ID[f"{key}/{kind.lower()}"]
        self.add(d, RDF.type, T[kind]); self.add(d, T["designates"], owner)
        self.record(ID[f"{key}/{kind.lower()}/record"], carries=d)
        self.add(ID[f"{key}/{kind.lower()}/record"], T["text"], Literal(text))
        return d

    def tz(self, user_record, key, value):
        z = ID[f"{key}/timezone"]
        self.add(z, RDF.type, T["TZ"]); self.add(user_record, T["uses_tz"], z)
        self.record(ID[f"{key}/timezone/record"], carries=z)
        self.add(ID[f"{key}/timezone/record"], T["text"], Literal(value))


def gtfs_datetime(service_day: datetime, hhmmss: str) -> datetime:
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return service_day + timedelta(hours=h, minutes=m, seconds=s)


def main():
    spec = load_spec()
    tzinfo = ZoneInfo(spec["service_timezone"])
    b = Builder()
    days = sorted({r["date"] for r in rows("calendar_dates.txt") if r["exception_type"] == "1"})
    if len(days) != 1:
        raise SystemExit(f"expected exactly one service day in calendar_dates.txt, found {days}")
    day = datetime.strptime(days[0], "%Y%m%d").replace(tzinfo=tzinfo)

    for a in rows("agency.txt"):
        org = ID[f"agency/{a['agency_id']}"]
        b.add(org, RDF.type, T["Org"]); b.add(org, RDFS.label, Literal(a["agency_name"]))
        b.designation("Name", org, f"agency/{a['agency_id']}", a["agency_name"])
        rec = b.record(ID[f"agency/{a['agency_id']}/record"], label=f"agency.txt row {a['agency_id']}")
        b.tz(rec, f"agency/{a['agency_id']}", a["agency_timezone"])
    for f in rows("feed_info.txt"):
        rec = b.record(ID[f"feed/{f['feed_version']}"], label=f"feed_info.txt {f['feed_version']}")
        b.designation("ArbId", rec, f"feed/{f['feed_version']}", f["feed_version"])
        b.tz(rec, f"feed/{f['feed_version']}", f["feed_timezone"])

    for s in rows("stops.txt"):
        key = f"station/{s['stop_id']}"
        st, loc = ID[key], ID[f"{key}/location"]
        b.add(st, RDF.type, T["RailStation"]); b.add(st, RDFS.label, Literal(s["stop_name"]))
        b.add(loc, RDF.type, T["GeoLoc"]); b.add(st, T["located_in"], loc)
        b.designation("Name", st, key, s["stop_name"])
        if s.get("stop_code"):
            b.designation("Code", st, key, s["stop_code"])
        b.designation("ArbId", st, key, s["stop_id"])
        cr = ID[f"{key}/coordinates"]
        b.add(cr, RDF.type, T["CoordRec"]); b.add(cr, T["is_about"], loc)
        rec = b.record(ID[f"{key}/coordinates/record"], carries=cr)
        b.add(rec, T["lat"], Literal(s["stop_lat"], datatype=XSD.decimal)); b.add(rec, T["lon"], Literal(s["stop_lon"], datatype=XSD.decimal))
        srec = b.record(ID[f"{key}/record"], label=f"stops.txt row {s['stop_id']}")
        b.add(srec, T["carrier_of"], ID[f"{key}/arbid"])
        if s.get("wheelchair_boarding", "") != "":
            b.add(srec, X.wheelchairBoardingCode, Literal(int(s["wheelchair_boarding"])))

    for r in rows("routes.txt"):
        key = f"route/{r['route_id']}"
        rt = ID[key]
        b.add(rt, RDF.type, T["Route"]); b.add(rt, RDFS.label, Literal(r["route_long_name"]))
        b.designation("Name", rt, key, r["route_long_name"]); b.designation("ArbId", rt, key, r["route_id"])
        rec = b.record(ID[f"{key}/record"], carries=rt, label=f"routes.txt row {r['route_id']}")
        b.add(rec, X.routeTypeCode, Literal(int(r["route_type"])))

    for t in rows("trips.txt"):
        key = f"trip/{t['trip_id']}"
        tr = ID[key]
        b.add(tr, RDF.type, T["Trip"]); b.add(tr, T["part_of"], ID[f"route/{t['route_id']}"])
        b.add(tr, RDFS.label, Literal(f"Train {t['trip_short_name']} to {t['trip_headsign']}"))
        b.designation("ArbId", tr, key, t["trip_id"])
        if t.get("trip_short_name"):
            b.designation("Code", tr, key, t["trip_short_name"])
        rec = b.record(ID[f"{key}/record"], carries=tr, label=f"trips.txt row {t['trip_id']}")
        b.add(rec, X.directionCode, Literal(int(t["direction_id"])))
        b.add(rec, X.headsignText, Literal(t["trip_headsign"]))
        if t.get("peak_offpeak", "") != "":
            b.add(rec, X.isPeakService, Literal(t["peak_offpeak"] == "1"))

    for s in rows("stop_times.txt"):
        key = f"trip/{s['trip_id']}/stop/{s['stop_sequence']}"
        sp = ID[key]
        b.add(sp, RDF.type, T["Stop"]); b.add(sp, T["part_of"], ID[f"trip/{s['trip_id']}"])
        b.add(sp, T["is_about"], ID[f"station/{s['stop_id']}"])
        rec = b.record(ID[f"{key}/record"], carries=sp)
        b.add(rec, X.stopSequence, Literal(int(s["stop_sequence"])))
        for which in ("arrival", "departure"):
            raw = s[f"{which}_time"]
            tod = ID[f"{key}/{which}"]
            b.add(tod, RDF.type, T["TOD"]); b.add(tod, T["part_of"], sp)
            b.add(tod, RDFS.label, Literal(f"{which} time"))
            tr = b.record(ID[f"{key}/{which}/record"], carries=tod)
            b.add(tr, T["text"], Literal(raw))
            try:
                b.add(tr, T["datetime"], Literal(gtfs_datetime(day, raw).isoformat(), datatype=XSD.dateTime))
            except ValueError:
                pass                                   # malformed time stays raw; QC reports the missing datetime

    b.g.serialize(INSTANCES, format="turtle")
    n = lambda c: len(set(b.g.subjects(RDF.type, T[c])))
    print(f"etl_gtfs: service day {days[0]}; {n('RailStation')} stations, {n('Route')} route, {n('Trip')} trips, "
          f"{n('Stop')} scheduled stops; {len(b.g)} triples -> data/instances.ttl")


if __name__ == "__main__":
    main()
