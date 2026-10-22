import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from onc_hydrophone_data.data import deployment_checker
from onc_hydrophone_data.data.deployment_checker import HydrophoneDeploymentChecker
from onc_hydrophone_data.data.downloader.deployment_workflows import (
    download_spectrograms_with_deployment_check,
    validate_deployment_coverage,
)


DEVICES = [
    {"deviceCode": "ICLISTENHF6324", "deviceId": 1, "deviceName": "Ocean Sonics icListen HF Hydrophone 6324"},
    {"deviceCode": "ICLISTENHF1354", "deviceId": 2, "deviceName": "Ocean Sonics icListen HF Hydrophone 1354"},
    {"deviceCode": "ICLISTENHF9999", "deviceId": 3, "deviceName": "Ocean Sonics icListen HF Hydrophone 9999"},
]

DEPLOYMENTS = [
    {
        "deviceCode": "ICLISTENHF1354",
        "locationCode": "BIIP",
        "begin": "2025-01-25T00:00:00.000Z",
        "end": None,
        "depth": 24.0,
        "lat": 49.301,
        "lon": -123.1125,
        "citation": {"doi": "10.34943/current-biip"},
    },
    {
        "deviceCode": "ICLISTENHF6324",
        "locationCode": "KEMFH.H1",
        "begin": "2023-09-08T00:00:00.000Z",
        "end": None,
        "depth": 2195.0,
        "lat": 47.9493,
        "lon": -129.0982,
        "citation": None,
    },
    {
        "deviceCode": "ICLISTENHF1354",
        "locationCode": "BIIP",
        "begin": "2022-10-04T00:00:00.000Z",
        "end": "2023-07-25T00:00:00.000Z",
        "depth": 26.0,
        "lat": 49.3007,
        "lon": -123.1111,
        "citation": {"doi": "10.34943/old-biip"},
    },
]

HIERARCHY = [
    {
        "locationCode": "ONC",
        "locationName": "Ocean Networks Canada",
        "children": [
            {
                "locationCode": "KEMF",
                "locationName": "Main Endeavour Field",
                "children": [
                    {
                        "locationCode": "KEMFH",
                        "locationName": "Hydrophone Array - Box Type",
                        "children": [
                            {"locationCode": "KEMFH.H1", "locationName": "Hydrophone A"},
                        ],
                    },
                ],
            },
            {"locationCode": "BIIP", "locationName": "Burrard Inlet Underwater Network"},
        ],
    },
]


class FakeONC:
    """Stand-in for the ONC client calls made by the deployment checker."""

    def __init__(self, deployments=DEPLOYMENTS, *, category_query_fails=False):
        self.deployments = deployments
        self.category_query_fails = category_query_fails
        self.deployment_queries = []

    def getDevices(self, filters):
        return [dict(device) for device in DEVICES]

    def getDeployments(self, filters):
        self.deployment_queries.append(filters)
        if "deviceCategoryCode" in filters:
            if self.category_query_fails:
                raise RuntimeError("category query rejected")
            return [dict(dep) for dep in self.deployments]
        return [dict(dep) for dep in self.deployments if dep["deviceCode"] == filters["deviceCode"]]

    def getLocations(self, filters):
        return [
            {"locationCode": "KEMFH.H1", "locationName": "Hydrophone A"},
            {"locationCode": "BIIP", "locationName": "Burrard Inlet Underwater Network"},
        ]

    def getLocationHierarchy(self, filters):
        return HIERARCHY


def make_checker(onc_client):
    with patch.object(deployment_checker, "ONC", return_value=onc_client):
        return HydrophoneDeploymentChecker("token")


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def test_overview_uses_one_query_and_adds_device_and_site_details():
    onc_client = FakeONC()
    overview = make_checker(onc_client).get_hydrophone_overview()

    assert onc_client.deployment_queries == [{"deviceCategoryCode": "HYDROPHONE"}]
    assert list(overview.columns) == list(deployment_checker.HYDROPHONE_OVERVIEW_COLUMNS)
    # Sorted by location code, then begin date.
    assert list(overview["location_code"]) == ["BIIP", "BIIP", "KEMFH.H1"]
    assert list(overview["begin_date"].dt.year) == [2022, 2025, 2023]
    assert list(overview["currently_deployed"]) == [False, True, True]
    assert list(overview["doi"].iloc[:2]) == ["10.34943/old-biip", "10.34943/current-biip"]

    array_row = overview.iloc[2]
    assert array_row["device_name"] == "Ocean Sonics icListen HF Hydrophone 6324"
    assert array_row["device_id"] == "1"
    assert array_row["location_name"] == "Main Endeavour Field"
    assert array_row["position_name"] == "Hydrophone A"
    assert array_row["depth_m"] == 2195.0


def test_overview_falls_back_to_one_query_per_device():
    onc_client = FakeONC(category_query_fails=True)
    overview = make_checker(onc_client).get_hydrophone_overview()

    assert {"deviceCode": "ICLISTENHF6324"} in onc_client.deployment_queries
    assert len(overview) == 3


def test_current_only_uses_begin_and_end_dates():
    now = datetime.now(timezone.utc)
    deployments = [
        {"deviceCode": "ICLISTENHF6324", "locationCode": "KEMFH.H1", "begin": iso(now - timedelta(days=10)),
         "end": iso(now + timedelta(days=10))},
        {"deviceCode": "ICLISTENHF1354", "locationCode": "BIIP", "begin": iso(now - timedelta(days=10)),
         "end": iso(now - timedelta(days=1))},
        {"deviceCode": "ICLISTENHF9999", "locationCode": "BIIP", "begin": iso(now + timedelta(days=1)),
         "end": None},
    ]
    overview = make_checker(FakeONC(deployments)).get_hydrophone_overview(current_only=True)

    assert list(overview["device_code"]) == ["ICLISTENHF6324"]


def test_print_current_overview(capsys):
    make_checker(FakeONC()).print_hydrophone_overview(current_only=True)
    output = capsys.readouterr().out
    lines = output.splitlines()

    assert lines[0].startswith("2 hydrophones deployed now at 2 sites (as of ")
    assert lines[2].split() == [
        "Location", "code", "Device", "code", "Begin", "End", "Depth", "(m)", "Latitude", "Longitude", "Location",
    ]
    assert lines[4].split() == [
        "BIIP", "ICLISTENHF1354", "2025-01-25", "ongoing", "24", "49.3010", "-123.1125",
        "Burrard", "Inlet", "Underwater", "Network",
    ]
    assert lines[5].endswith("Main Endeavour Field (Hydrophone A)")
    assert "nan" not in output
    assert "Not listed" not in output


def test_print_full_overview_counts_undeployed_hydrophones(capsys):
    make_checker(FakeONC()).print_hydrophone_overview()
    output = capsys.readouterr().out

    assert output.startswith("3 deployments of 2 hydrophones at 2 sites; 2 deployed now (as of ")
    assert "2022-10-04  2023-07-25" in output
    assert "Not listed: 1 registered hydrophone with no deployment records." in output


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            ("Endeavour", "Main Endeavour Field", "Hydrophone Array - Box Type", "Hydrophone A"),
            ("Main Endeavour Field", "Hydrophone A"),
        ),
        (("Cascadia Basin", "ODP 1026", "Hydrophone Broadband"), ("ODP 1026", "Hydrophone Broadband")),
        (
            ("Strait of Georgia East", "Underwater Listening Station - Secondary", "JASCO Hydrophone 1"),
            ("Underwater Listening Station - Secondary", "JASCO Hydrophone 1"),
        ),
        (("Burrard Inlet", "Burrard Inlet Underwater Network"), ("Burrard Inlet Underwater Network", None)),
        (("Fraser River Delta", "Seismic Liquefaction In-Situ Penetrometer "), ("Seismic Liquefaction In-Situ Penetrometer", None)),
        ((), ("Hydrophone A", "Hydrophone A")),
    ],
)
def test_display_location_names_the_site_and_array_position(path, expected):
    checker = make_checker(FakeONC())
    assert checker._resolve_display_location("Hydrophone A", path) == expected


def _downloader_with_deployments(deployments):
    downloader = SimpleNamespace(
        logger=logging.getLogger("test"),
        _get_cached_deployments=lambda: deployments,
        download_spectrograms_with_sampling_schedule=MagicMock(),
    )
    downloader.validate_deployment_coverage = lambda *args: validate_deployment_coverage(downloader, *args)
    return downloader


def test_deployment_check_samples_a_deployed_period():
    deployment = SimpleNamespace(
        device_code="ICLISTENHF6324",
        begin_date=datetime(2023, 9, 8, tzinfo=timezone.utc),
        end_date=None,
        location_name="Main Endeavour Field",
    )
    downloader = _downloader_with_deployments([deployment])

    sampled = download_spectrograms_with_deployment_check(
        downloader, "ICLISTENHF6324", (2024, 4, 1), 10, num_days=3, filetype="mat", spectrograms_per_batch=2,
    )

    assert sampled is True
    downloader.download_spectrograms_with_sampling_schedule.assert_called_once_with(
        "ICLISTENHF6324", (2024, 4, 1), 10, num_days=3, filetype="mat", spectrograms_per_batch=2, download_audio=False,
    )


def test_deployment_check_skips_undeployed_period(capsys):
    deployment = SimpleNamespace(
        device_code="ICLISTENHF6324",
        begin_date=datetime(2023, 9, 8, tzinfo=timezone.utc),
        end_date=None,
        location_name="Main Endeavour Field",
    )
    downloader = _downloader_with_deployments([deployment])

    sampled = download_spectrograms_with_deployment_check(
        downloader, "ICLISTENHF6324", (2020, 1, 1), 10, num_days=3,
    )

    assert sampled is False
    downloader.download_spectrograms_with_sampling_schedule.assert_not_called()
    assert "2023-09-08 to ongoing at Main Endeavour Field" in capsys.readouterr().out
