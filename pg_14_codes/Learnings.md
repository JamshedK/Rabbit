# Learnings & Changes

## Issues Encountered and Fixed

### 1. Missing Directory Error (FileNotFoundError)
**Date:** February 2, 2026

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: './knowledge/postgres/suggested_knobs_value/suggested_knobs_value_tpch_task1.json'
```

**Root Cause:**
The code tried to write to a JSON file in a directory that didn't exist.

**Solution:**
Created the missing directory:
```bash
mkdir -p /home/karimnazarovj/Rabbit/knowledge/postgres/suggested_knobs_value
```

**Note:** The JSON file itself should NOT be created manually as empty. The code generates and writes the content automatically.

---

### 2. Empty Transfer Learning Data Error (IndexError)
**Date:** February 2, 2026

**Error:**
```
IndexError: list index out of range
```
Location: `/home/karimnazarovj/Rabbit/space_optimizer/coarse_space.py` line 33

**Root Cause:**
The code attempted to access the last element `[-1]` of an empty list when no historical transfer learning data was available. The file `knowledge/postgres/incumbents_transfer_tpch.json` contained an empty list `[]`.

**Solution:**
Modified [coarse_space.py](../space_optimizer/coarse_space.py) to handle empty transfer data gracefully:

---

### 3. Empty JSON File Error (JSONDecodeError)
**Date:** February 2, 2026

**Error:** `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` at [coarse_space.py](../space_optimizer/coarse_space.py) line 44

**Root Cause:** Manually created empty `suggested_knobs_value_tpch_task1.json`. Code skipped generation because file existed, then failed parsing empty JSON.

**Solution:** Delete manually created file. Let code generate JSON with proper content.

---

### 4. Hardcoded MySQL Path & Missing extra_knobs_configs File
**Date:** February 2, 2026

**Error:** `FileNotFoundError: './knowledge/mysql/extra_knobs_configs_tpch.json'` when running postgres

**Root Cause:** Two issues:
1. Line 44 in [run.py](../run.py) hardcoded path to `mysql` instead of using `{args.db}`
2. When no transfer learning data exists, `extra_knobs_configs_path` file wasn't created but code tries to read it

**Solution:** Modified [run.py](../run.py):
1. Changed `f"./knowledge/mysql/extra_knobs_configs_{args.test}.json"` to `f"./knowledge/{args.db}/extra_knobs_configs_{args.test}.json"`
2. Added creation of empty `extra_knobs_configs` file (`{}`) when no transfer data (lines 107-109)

---

## Notes

- System can run without historical transfer learning data
- When no transfer data exists, system uses default configurations
- **Only create directories, never empty JSON files** - let code generate content
- Directory structure must exist before code writes JSON files
