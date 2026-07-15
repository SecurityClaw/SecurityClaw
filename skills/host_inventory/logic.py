from core.endpoint_security import collect_inventory


def run(context: dict) -> dict:
    return collect_inventory()
