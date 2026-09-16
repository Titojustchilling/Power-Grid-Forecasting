from __future__ import annotations

import pandas as pd


def run_power_flow(forecast_df: pd.DataFrame, results_path):
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError as e:
        raise RuntimeError("pandapower is required for grid integration") from e

    net = pn.case14()

    # IEEE-14 base demand.
    base_load = float(net.load.p_mw.sum())

    # Use the forecast relative to its own recent baseline rather than
    # applying the PJM RTO's absolute MW value to the tiny IEEE-14 system.
    forecast_mean = float(forecast_df["load_mw"].mean())

    recent_baseline = float(
        forecast_df["load_mw"].iloc[0]
    )

    if recent_baseline > 0:
        load_change_pct = (forecast_mean / recent_baseline) - 1.0
    else:
        load_change_pct = 0.0

    # Apply the same percentage change to the IEEE-14 benchmark.
    scale = 1.0 + load_change_pct

    net.load["p_mw"] *= scale
    net.load["q_mvar"] *= scale

    # Run AC power flow.
    pp.runpp(net, init="flat", max_iteration=50)

    result = {
        "forecast_mean_load_mw": forecast_mean,
        "ieee14_base_load_mw": base_load,
        "load_change_pct": load_change_pct * 100,
        "ieee14_load_scale_factor": scale,
        "min_bus_voltage_pu": float(net.res_bus.vm_pu.min()),
        "max_line_loading_pct": float(net.res_line.loading_percent.max()),
        "active_power_losses_mw": float(net.res_line.pl_mw.sum()),
    }

    pd.DataFrame([result]).to_csv(results_path, index=False)

    return result