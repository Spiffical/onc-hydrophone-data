# Deployments & Inventory

Use the deployment inventory to pick devices and valid dates before downloading data.

![Inventory table example](assets/inventory_table.svg){: width="100%" }

```python
from onc_hydrophone_data.data.deployment_checker import HydrophoneDeploymentChecker

checker = HydrophoneDeploymentChecker(ONC_TOKEN)
inventory = checker.collect_hydrophone_inventory()

# Current deployments (active devices)
checker.show_hydrophone_inventory_table(inventory, view="current")

# Deployment history (all deployments)
checker.show_hydrophone_inventory_table(inventory, view="history", max_rows=20)
```

Filter to specific devices after reviewing the tables:

```python
checker.show_device_deployments(device_codes=["ICLISTENHF6324"], inventory=inventory)
# Or by numeric IDs if you have them:
# checker.show_device_deployments(device_ids=[12345], inventory=inventory)
```

## Visual availability

Get a deployment-aware view of data availability for a specific hydrophone:

```python
from onc_hydrophone_data.data.deployment_checker import HydrophoneDeploymentChecker
from onc_hydrophone_data.utils import (
    plot_deployment_availability_timeline,
    plot_availability_calendar,
)

checker = HydrophoneDeploymentChecker(ONC_TOKEN)
availability = checker.get_device_availability("ICLISTENHF6324", bin_size="day")

# Timeline view (deployments on rows, data gaps highlighted)
plot_deployment_availability_timeline(availability)

# Calendar view (daily coverage heatmap)
plot_availability_calendar(availability)

# Convenience wrapper
# checker.plot_device_availability("ICLISTENHF6324", style="timeline")
```

!!! note
    Availability uses ONC archive file listings, so long date ranges may take a
    while to query. Limit the date range or adjust `max_days_per_request` if needed.

!!! tip
    The inventory tables already include deployment start/end dates, so you can
    choose valid time ranges directly from Table 1–3.
