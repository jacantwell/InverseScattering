from .base import Field
from .scattered import ScatteredField


def simulation_factory(simulation_config: dict) -> Field:
    """
    This function creates a field object of the specified type.

    Parameters:
    simulation_config: dict
        A dictionary containing the configuration of the simulation. Must contain the key 'simulation_type', which specifies
        the type of simulation to instansiate. Additional keys are passed as keyword arguments to the simulation constructor.

    Returns:
    Field
        The field object of the specified type.
    """

    field_config = simulation_config.copy()
    simulation_type = field_config.pop("simulation_type", None)

    if simulation_type == "scattered_field":
        return ScatteredField(**field_config)
    else:
        raise ValueError(f"Simulation type not recognised/found")
