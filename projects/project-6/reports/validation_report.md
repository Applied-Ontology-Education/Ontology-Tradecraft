# Project 6 validation report

Ontology 1919 triples (extension + CCO slim), instances 27494 triples.

| Check | Result |
|---|---|
| ELK (ROBOT) | consistent, no unsatisfiable classes (1.5 s) |
| HermiT (ROBOT) | consistent (4.1 s) |
| SHACL (9 node shapes) | 0 findings, 0 unexpected |
| SPARQL QC (10 queries) | 1 findings, 0 unexpected |
| QC checks present | yes |
| **Overall** | **PASS** |

## Known (acknowledged) source-data issues
- `feed_info.txt` `feed_timezone` = `America/New York`: expected America/New_York (valid IANA zone; agency.txt uses it). feed_timezone is also not a GTFS Schedule field (LIRR extension).

## SPARQL QC detail

- `human_01_invalid_timezone.rq` (human): 1 findings (0 unexpected)
- `qc_01_duplicate_stop_sequence.rq` (llm): 0 findings (0 unexpected)
- `qc_02_stop_station_missing.rq` (llm): 0 findings (0 unexpected)
- `qc_03_trip_route_missing.rq` (llm): 0 findings (0 unexpected)
- `qc_04_arrival_after_departure.rq` (llm): 0 findings (0 unexpected)
- `qc_05_stop_time_order_vs_sequence.rq` (llm): 0 findings (0 unexpected)
- `qc_06_time_text_format.rq` (llm): 0 findings (0 unexpected)
- `qc_07_time_datetime_mismatch.rq` (llm): 0 findings (0 unexpected)
- `qc_08_code_list_values.rq` (llm): 0 findings (0 unexpected)
- `qc_09_coordinate_range.rq` (llm): 0 findings (0 unexpected)
