# Project 6 — Windows output encoding fix

## Cause

`validate_manifest_abox.py` printed a Unicode right arrow (`→`) to its process stdout before launching ROBOT extraction. When the process stdout is encoded with Windows cp1252, this triggers `UnicodeEncodeError` and the validation stage fails without running HermiT. This update changes **only** that log separator to ASCII `->`.

## Installation

Run from `C:\Users\cappa\Ontology-Tradecraft\projects\project-6` with `.venv-mowl` activated:

```powershell
Copy-Item .\scripts\validate_manifest_abox.py .\scripts\validate_manifest_abox_before_encoding_fix.py
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Project6_Manifest_Windows_Encoding_Fix.zip" -DestinationPath . -Force
python -m unittest discover -s scripts -p "test_validate_manifest_abox.py" -v
```

## Validate the CURRENT artifacts first (no Grok rerun)

```powershell
python -u scripts/validate_manifest_abox.py --project-root . --workbook templates/passenger_manifest_template.xlsx --strict-occurrence --timeout 300 --hard-timeout 1200
Get-Content .\reports\manifest_semantic_validation.json
```

This reuses the **currently generated** augmented ontology and ABox files. The SHACL and HermiT checks still execute; it does not reuse a cached validation report. It does not rerun the costly live Grok/mOWL pipeline.

## Optional full end-to-end verification

To obtain an all-PASS `step8_run_summary.json`, run the complete `python -u scripts/run_all.py --manifest templates/passenger_manifest_template.xlsx --flight-document FL-20261117-AA101=FPDOC-AA101` command after the standalone validator passes. This makes fresh Grok requests, regenerates intermediates, and archives the previous outputs.

Original `src/module.ttl` and augmented `generated/module_augmented.ttl` are never changed by the standalone validator.
