"""Render-path benchmarks.

Run in this order:

    verify     -- prove the shared renderer is pixel-identical (do this first)
    baseline   -- measure the current one-renderer-per-env path
    calibrate  -- sweep M and pick the number of shared renderers
    compare    -- old vs new, interleaved, with plots
"""
