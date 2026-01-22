"""
Location Mappings for Hydrophone Divert Monitoring

Maps email location names from divert notifications to ONC hydrophone location codes.
Adapted from the hydrophonedashboard repository for shared use.
"""

import re
from typing import Dict, Iterable, List, Set

# Location mapping - maps email location names to hydrophone location codes
# Based on actual ONC hydrophone deployments and divert system naming
LOCATION_MAPPING = {
    # ================================
    # SoG DDS (Strait of Georgia) System Locations
    # ================================
    'SoG_East': ['ECHO3.H1', 'ECHO3.H2', 'ECHO3.H3', 'ECHO3.H4'],  # SoG East = ECHO3 array
    'SoG_Delta': [],  # SoG Delta - no active hydrophones mapped
    'SoG_Central': ['PSGCH.H1', 'PSGCH.H3'],  # SoG Central = PSGCH array (only H1 and H3 active)

    # ================================
    # Saanich DDS System Locations
    # ================================
    'Saanich_Inlet': ['PVIPH.H1', 'PVIPH.H3'],  # Saanich Inlet = PVIPH array (only H1 and H3 active)

    # ================================
    # NC-DDS (Northern Canadian) System Locations - with bracket format
    # ================================
    '[1] Barkley Cnyn': ['BACNH.H1', 'BACNH.H2', 'BACNH.H3', 'BACNH.H4', 'BACUS'],
    '[2] ODP 1027': ['CBCH.H1', 'CBCH.H2', 'CBCH.H3', 'CBCH.H4'],
    '[3] Endeavour': ['KEMFH.H1', 'KEMFH.H2', 'KEMFH.H3', 'KEMFH.H4'],
    '[4] ODP 889': ['CQSH.H1', 'CQSH.H2', 'CQSH.H3', 'CQSH.H4'],
    '[5] Folger Pass': ['FGPD'],

    # ================================
    # Additional ODP Sites (Ocean Drilling Program)
    # ================================
    'ODP 1364A': ['CQSH.H1', 'CQSH.H2', 'CQSH.H3', 'CQSH.H4'],
    'ODP 1026': ['NC27.H3', 'NC27.H4'],

    # ================================
    # NC-DDS System Locations - without bracket format (fallback)
    # ================================
    'Barkley Cnyn': ['BACNH.H1', 'BACNH.H2', 'BACNH.H3', 'BACNH.H4', 'BACUS'],
    'ODP 1027': ['CBCH.H1', 'CBCH.H2', 'CBCH.H3', 'CBCH.H4'],
    'Endeavour': ['KEMFH.H1', 'KEMFH.H2', 'KEMFH.H3', 'KEMFH.H4'],
    'ODP 889': ['CQSH.H1', 'CQSH.H2', 'CQSH.H3', 'CQSH.H4'],
    'Folger Pass': ['FGPD'],

    # ================================
    # Additional Location Mappings (if these locations appear in emails)
    # ================================
    'Burrard Inlet': ['BIIP'],
    'Cambridge Bay': ['CBYIP'],
    'China Creek': ['CCIP'],
    'Clayoquot Slope': ['CQSH.H1', 'CQSH.H2', 'CQSH.H3', 'CQSH.H4'],
    'Cascadia Basin (ODP 1027)': ['CBCH.H1', 'CBCH.H2', 'CBCH.H3', 'CBCH.H4'],
    'Digby Island': ['DIIP'],
    'Hartley Bay': ['HBIP'],
    'Holyrood Bay': ['HRBIP'],
    'Kitamaat Village': ['KVIP'],
}

SOG_DDS_LOCATIONS = {
    'SoG_East', 'SoG_Delta', 'SoG_Central'
}

SAANICH_DDS_LOCATIONS = {
    'Saanich_Inlet'
}

NC_DDS_LOCATIONS = {
    '[1] Barkley Cnyn', '[2] ODP 1027', '[3] Endeavour', '[4] ODP 889', '[5] Folger Pass',
    'Barkley Cnyn', 'ODP 1027', 'Endeavour', 'ODP 889', 'Folger Pass'
}

ODP_LOCATIONS = {
    'ODP 1027', 'ODP 1364A', 'ODP 1026', 'ODP 889',
    '[2] ODP 1027', '[4] ODP 889'
}


def get_system_for_location(location_name: str) -> str:
    if location_name in SOG_DDS_LOCATIONS:
        return 'SoG DDS'
    if location_name in NC_DDS_LOCATIONS:
        return 'NC-DDS'
    if location_name in SAANICH_DDS_LOCATIONS:
        return 'Saanich DDS'
    return 'Unknown'


def get_hydrophone_codes(location_name: str) -> List[str]:
    return LOCATION_MAPPING.get(location_name, [])


def is_odp_location(location_name: str) -> bool:
    return location_name in ODP_LOCATIONS


def get_all_mapped_locations() -> List[str]:
    return [location for location, codes in LOCATION_MAPPING.items() if codes]


def get_unmapped_locations() -> List[str]:
    return [location for location, codes in LOCATION_MAPPING.items() if not codes]


def build_reverse_location_mapping() -> Dict[str, List[str]]:
    code_to_names: Dict[str, Set[str]] = {}
    for location_name, codes in LOCATION_MAPPING.items():
        for code in codes:
            code_to_names.setdefault(code, set()).add(location_name)
    return {code: sorted(names) for code, names in code_to_names.items()}


def normalize_mapping_label(label: str) -> str:
    label = re.sub(r'^\[\d+\]\s*', '', label).strip()
    label = label.replace('_', ' ')
    label = label.replace('Cnyn', 'Canyon')
    return label.strip()


def build_friendly_mapping_names(raw_names: Iterable[str]) -> List[str]:
    cleaned = [normalize_mapping_label(name) for name in raw_names if name]
    cleaned = sorted(set(name for name in cleaned if name))
    if not cleaned:
        return []
    lowered = [name.lower() for name in cleaned]
    result = []
    for idx, name in enumerate(cleaned):
        is_substring = any(
            idx != other
            and lowered[idx] in lowered[other]
            and len(lowered[idx]) < len(lowered[other])
            for other in range(len(cleaned))
        )
        if not is_substring:
            result.append(name)
    return result
