# Comprehensive Sweep Campaign Report

**Timestamp**: 20260428-090746
**Results Directory**: sweep_results\campaign_20260428-090746

## Campaign Configuration

- **Profiles**: paper-baseline, operational, engineering-screen, resilience
- **Total Sweeps**: 8

## Sweep Results

| Sweep | Status | Duration (s) | Return Code |
|-------|--------|-------------|-------------|
| LOB-Scaling | SUCCESS | 1.31 | 0 |
| Sensitivity | SUCCESS | 6.14 | 0 |
| T3-Default | FAILED | 6.68 | 1 |
| Stream-Balance | FAILED | 4.30 | 1 |
| Mission-Scenarios | FAILED | 0.17 | 1 |
| T3-HighRes | SUCCESS | 22.45 | 0 |
| T1-Default | FAILED | 28214.71 | 4294967295 |
| T1-HighRes | FAILED | 28213.41 | 4294967295 |

**Total Duration**: 56469.17s (941.15 minutes)
**Success Rate**: 3/8 (37.5%)

## Output Files

The following files were generated:

- `lob_scaling.log` - Execution log
- `sensitivity.log` - Execution log
- `t3_highres.log` - Execution log

## Next Steps

1. Review individual sweep logs for detailed results
2. Examine generated plots (PNG files)
3. Compare default vs high-resolution results
4. Aggregate raw data for analysis
