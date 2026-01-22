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

!!! tip
    The inventory tables already include deployment start/end dates, so you can
    choose valid time ranges directly from Table 1–3.
