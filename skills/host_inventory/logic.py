from skills.endpoint_telemetry.logic import collect_inventory


def run(context: dict) -> dict:
    return collect_inventory()
