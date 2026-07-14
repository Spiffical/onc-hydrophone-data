# 2. Find a Hydrophone and Valid Dates

A **device code** identifies an instrument, while a **deployment** tells you
where and when that instrument was installed. Always confirm both before
downloading: a valid device code can still return no files for dates outside a
deployment or during an archive gap.

## List current and historical deployments

```python
from onc_hydrophone_data.data.deployment_checker import (
    HydrophoneDeploymentChecker,
)
from onc_hydrophone_data.onc.common import load_config

onc_token, _ = load_config()
checker = HydrophoneDeploymentChecker(onc_token)
inventory = checker.collect_hydrophone_inventory()

# Instruments that appear to be deployed now.
checker.show_hydrophone_inventory_table(inventory, view="current")

# Historical deployments, including completed ones.
checker.show_hydrophone_inventory_table(
    inventory,
    view="history",
    max_rows=20,
)
```

For example, these are real records for the four-device hydrophone array at
Main Endeavour Field:

![Current ONC hydrophone-array deployments at Main Endeavour Field](assets/figures/deployment_inventory.webp){: width="100%" loading="lazy" }

*Source: [ONC Hydrophone Location Codes & Data Types](https://wiki.oceannetworks.ca/pages/viewpage.action?pageId=72548654),
accessed 2026-07-14. An empty ONC end date means the deployment was ongoing in
that source snapshot.*

Focus on these columns:

| Column | Why it matters |
| --- | --- |
| `device_code` | The value passed to download methods |
| `location_name` | Where the hydrophone was deployed |
| `begin_date` / `end_date` | The time interval you can sensibly query |
| `position_name` | The element within a multi-hydrophone array, when present |

Once you find a candidate, show only that instrument's deployment history:

```python
checker.show_device_deployments(
    device_codes=["ICLISTENHF6324"],
    inventory=inventory,
)
```

The plot below uses real ONC deployment records from Main Endeavour Field. The
older single-hydrophone deployments are followed by the four-position array
that includes `ICLISTENHF6324`.

![Real ONC hydrophone deployment history at Main Endeavour Field](assets/figures/deployment_timeline.webp){: width="100%" loading="lazy" }

Grey bars are completed deployments. Teal bars had no end date in the
[official ONC inventory](https://wiki.oceannetworks.ca/pages/viewpage.action?pageId=72548654)
when accessed on 2026-07-14. The legend sits below the time axis so it never
covers a deployment bar.

## Confirm archive availability

A deployment window means the instrument was installed; it does not guarantee
that every hour has an archived audio file. Query a modest date range and plot
daily coverage:

```python
availability = checker.get_device_availability(
    "ICLISTENHF6324",
    start_date="2024-04-01",
    end_date="2024-07-01",
    timezone_str="UTC",
    bin_size="day",
)
```

!!! note
    Availability queries inspect ONC archive listings. Start with weeks or a
    few months rather than the instrument's entire history.

### Timeline view

```python
from onc_hydrophone_data.utils import plot_deployment_availability_timeline

plot_deployment_availability_timeline(availability)
```

Each row is a deployment. Green segments have archived data; red segments are
gaps within a deployment; blank space separates deployment windows. The
percentage at right summarizes bins with some data. The legend is placed below
the axes so it does not obscure any interval.

### Calendar view

```python
from onc_hydrophone_data.utils import plot_availability_calendar

plot_availability_calendar(availability)
```

Dark green days have high coverage, lighter green days are partial, red days
have no data during a deployment, and grey days are outside deployment windows.
Its legend also sits below the calendar, while the continuous coverage scale
stays to the right.

!!! info "Why there is no frozen availability screenshot"
    Archive coverage is live, date-specific ONC data and requires an
    authenticated query. Rather than publish invented gaps or let a screenshot
    become stale, this page shows real deployment records and has you generate
    the two availability plots from your own current ONC query.

## Carry your choice into the download

Write down:

1. the `device_code`;
2. a UTC start time inside a green interval;
3. a UTC end time a few minutes later for your first test.

Continue to **[3. Download Audio and Make a Spectrogram](quickstart.md)** with
those values.
