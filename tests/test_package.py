def test_package_exposes_version() -> None:
    from shipment_triage import __version__

    assert __version__ == "0.1.0"
