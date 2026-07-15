from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from onc_hydrophone_data.utils import (
    plot_availability_calendar,
    plot_deployment_availability_timeline,
)


def _daily_availability() -> dict:
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=14)
    deployment = SimpleNamespace(
        device_code="ICLISTENHF6324",
        location_name="Main Endeavour Field",
        location_code="KEMFH.H1",
        position_name="A",
        begin_date=start,
        end_date=None,
    )
    bins = [
        {
            "start": start + timedelta(days=index),
            "end": start + timedelta(days=index + 1),
            "coverage": 1.0,
            "status": "data",
            "deployment_index": 0,
        }
        for index in range(14)
    ]
    return {
        "device_code": "ICLISTENHF6324",
        "bin_size": "day",
        "start": start,
        "end": end,
        "deployments": [deployment],
        "bins": bins,
        "deployment_summary": [
            {"deployment_index": 0, "coverage_ratio": 1.0}
        ],
    }


def _assert_legend_is_below_plot(fig, ax) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bounds = fig.legends[0].get_window_extent(renderer)
    axes_bounds = ax.get_window_extent(renderer)
    tolerance_pixels = 2.0
    assert legend_bounds.y1 <= axes_bounds.y0 + tolerance_pixels


def test_timeline_legend_does_not_cover_plot() -> None:
    fig, ax = plot_deployment_availability_timeline(
        _daily_availability(),
        show=False,
    )
    try:
        _assert_legend_is_below_plot(fig, ax)
    finally:
        plt.close(fig)


def test_calendar_legend_does_not_cover_plot() -> None:
    fig, ax = plot_availability_calendar(_daily_availability(), show=False)
    try:
        _assert_legend_is_below_plot(fig, ax)
    finally:
        plt.close(fig)


def test_calendar_excludes_the_query_end_boundary() -> None:
    fig, ax = plot_availability_calendar(_daily_availability(), show=False)
    try:
        calendar_grid = ax.images[0].get_array()
        assert calendar_grid.shape[:2] == (7, 2)
    finally:
        plt.close(fig)


def test_calendar_handles_a_missing_last_bin_start() -> None:
    availability = _daily_availability()
    availability["bins"][-1]["start"] = None

    assert plot_availability_calendar(availability, show=False) is None
