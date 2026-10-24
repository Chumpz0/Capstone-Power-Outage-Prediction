"""Shared geography and chronological split definitions; no data access."""
SOUTHEAST_STATES = {
    'ALABAMA': ('01', 'AL'), 'FLORIDA': ('12', 'FL'), 'GEORGIA': ('13', 'GA'),
    'NORTH CAROLINA': ('37', 'NC'), 'SOUTH CAROLINA': ('45', 'SC'), 'TENNESSEE': ('47', 'TN'),
}
SPLIT_YEARS = {'train': (None, 2023), 'validation': (2024, 2024), 'test': (2025, 2025)}
SPLIT_BOUNDARIES = ('2024-01-01', '2025-01-01', '2026-01-01')
