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

### 5. Dimension Mismatch in Second Stage (RuntimeError)
**Date:** February 2, 2026

**Error:**
```
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 20 but got size 23 for tensor number 1 in the list.
```
Location: BoTorch optimization in [second_stage.py](../config_recommender/second_stage.py) during GP model training

**Root Cause:**
The `select_add_knobs()` method incremented `self.target_knobs_num` from 20 to 23, but `change_search_space()` skipped all 3 knobs (not in `extra_knobs_space` due to missing historical data). This created a mismatch:
- GP model trained on 20-dimensional historical data
- Optimization bounds created with 23 dimensions (incorrect `target_knobs_num`)

**Solution:**
Modified [second_stage.py](../config_recommender/second_stage.py):
1. Removed premature increment in `select_add_knobs()` - don't increment counter until knob is actually added
2. In `change_search_space()`, remove skipped knobs from `self.target_knobs` list
3. Only increment `self.target_knobs_num` AFTER successful addition to search space (line 320)

This ensures `self.target_knobs_num` always matches the actual hyperparameter count in `self.search_space`.

---

## Important Configuration Notes

### Performance Metric Selection
**For TPC-H workloads:** Make sure to change the performance metric in [run.py](../run.py) line 35:

```python
# For TPC-H (OLAP workload), optimize for latency:
performance_metric = ['-lat']  # NOT ['tps']

# For TPC-C (OLTP workload), optimize for throughput:
performance_metric = ['tps']
```

**Why:** TPC-H is an analytical (OLAP) workload where faster query execution (lower latency) is the goal. TPC-C is transactional (OLTP) where higher throughput (transactions per second) matters more.

---

## Why So Many API Calls??

**FirstStage** is where all the juice happens. Uses `_get_next_point_hybrid()` to generate 10 candidate configs (5 from GP + 5 from LLM). Then `_get_next_point_llm()` evaluates each of those 10 configs by calling the LLM 5 times for ensemble predictions (k=5). 

**Total per iteration: 51 LLM calls**
- 1 call to generate 5 LLM candidate configs
- 50 calls to evaluate all 10 candidates (10 configs × 5 predictions each)

**Result:** Only 1 out of 10 evaluated configs gets tested on the actual database. The other 9 are discarded.

**Cost Impact:**
- With GPT-4o: ~$21 per FirstStage run (~1,275 total calls)
- With GPT-4o-mini: ~$0.50 per FirstStage run (40x cheaper)

**Potential Optimizations:**
- Reduce k from 5 to 2 (60% cost reduction)
- Reduce candidate_nums from 5 to 3 (fewer candidates to evaluate)
- Switch to GPT-4o-mini in `.env` file


## Notes

- System can run without historical transfer learning data
- When no transfer data exists, system uses default configurations
- **Only create directories, never empty JSON files** - let code generate content
- Directory structure must exist before code writes JSON files
- **Match performance metric to workload type**: `-lat` for OLAP (TPC-H), `tps` for OLTP (TPC-C)
