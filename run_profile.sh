/opt/intel/oneapi/vtune/latest/bin64/vtune --collect gpu-hotspots \
    pytest tests/test_layernorm.py::test_rms_norm_profiling


# /opt/intel/oneapi/vtune/latest/bin64/vtune --collect gpu-hotspots \
#     pytest tests/test_topk_per_row.py::test_top_k_per_row_profiling
