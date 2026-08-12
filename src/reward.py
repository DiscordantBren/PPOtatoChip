from .netlist import Netlist


def compute_hpwl(netlist: Netlist, config: dict[str, tuple[int, int]], num_rows: int, num_cols: int) -> float:

    cell_width = netlist.canvas["width"] / num_cols
    cell_height = netlist.canvas["height"] / num_rows

    total_hpwl = 0.0

    for net_id, node_ids in netlist.nets.items():
        x_coords = []
        y_coords = []

        for node_id in node_ids:
            if node_id not in config:
                raise ValueError(
                    f"Node {node_id} from net {net_id} is missing from config"
                )

            row, col = config[node_id]

            x = (col + 0.5) * cell_width
            y = (row + 0.5) * cell_height

            x_coords.append(x)
            y_coords.append(y)

        net_hpwl = (
            max(x_coords) - min(x_coords)
            + max(y_coords) - min(y_coords)
        )

        total_hpwl += net_hpwl

    return total_hpwl