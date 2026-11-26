"""Optional QC: submit src/module_augmented.ttl to the OOPS! pitfall scanner (web service) -> reports/oops_report.xml.

Never fails the pipeline: if the service is unreachable the step records "skipped". OOPS! sees only the
module's own axioms (imports are not sent)."""
import json
import re
import sys
import urllib.request
from pathlib import Path

from rdflib import Graph

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS, SRC, write_json  # noqa: E402

URL = "https://oops.linkeddata.es/rest"


def main():
    rdfxml = Graph().parse(SRC / "module_augmented.ttl").serialize(format="xml")
    body = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?><OOPSRequest><OntologyURI></OntologyURI>"
            f"<OntologyContent><![CDATA[{rdfxml}]]></OntologyContent><Pitfalls></Pitfalls>"
            "<OutputFormat>XML</OutputFormat></OOPSRequest>").encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/xml"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            xml = r.read().decode("utf-8", "replace")
        (REPORTS / "oops_report.xml").write_text(xml)
        codes = sorted(set(re.findall(r"P\d{2}", " ".join(re.findall(r"<oops:Code>(.*?)</oops:Code>", xml)))))
        status = {"status": "ok", "pitfall_codes": codes, "report": "reports/oops_report.xml"}
    except Exception as e:  # service down, offline, blocked
        status = {"status": f"skipped ({type(e).__name__}: {str(e)[:120]})"}
    m = json.loads((REPORTS / "mowl_metrics.json").read_text()); m["oops"] = status
    write_json(REPORTS / "mowl_metrics.json", m)
    print(f"oops_check: {status['status']}" + (f"; pitfalls {', '.join(status['pitfall_codes']) or 'none'}" if status["status"] == "ok" else ""))


if __name__ == "__main__":
    main()
