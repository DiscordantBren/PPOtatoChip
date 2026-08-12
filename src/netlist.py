import json

# Validation function for netlist json

def validate_netlist_data(netlist_dict: dict) -> None:
    required_keys = {'name', 'canvas', 'blocks', 'nets'}
    missing_keys = required_keys - netlist_dict.keys()
    
    # Checks if the required keys are present
    if len(missing_keys) != 0:
        raise ValueError(f"Missing required keys: {missing_keys}")

    # Checks if block ids are unique
    block_ids = [block['id'] for block in netlist_dict['blocks']]
    if len(block_ids) != len(set(block_ids)):      
        raise ValueError("Duplicate block IDs found")

    # Checks if net ids are unique
    net_ids = [net['net_id'] for net in netlist_dict['nets']]
    if len(net_ids) != len(set(net_ids)):       
        raise ValueError("Duplicate net IDs found")
    
    # Checks if canvas dims are positive
    if netlist_dict['canvas']['width'] <= 0 or netlist_dict['canvas']['height'] <= 0:       
        raise ValueError("Non-positive canvas dimension")
    
    # Checks if block dims are positive
    for block in netlist_dict['blocks']:        
        if block['width'] <= 0 or block['height'] <= 0:
            raise ValueError(f"Non-positive dimension in {block['id']}")
        
    # Checks if blocks fit inside canvas   
    for block in netlist_dict['blocks']:
        if block['width'] > netlist_dict['canvas']['width'] or block['height'] > netlist_dict['canvas']['height']:
            raise ValueError(f"{block['id']} does not fit inside canvas")
        
    # Checks if block ids referenced in a net are valid
    for net in netlist_dict['nets']:
        for block_id_ref in net['blocks']:
            if block_id_ref not in block_ids:
                raise ValueError(f"{block_id_ref} is not a valid block id")

    


class Netlist:
    def __init__(self, netlist_path: str) -> None:
        with open(netlist_path) as f:
            data = json.load(f)

        validate_netlist_data(data)

        self.name = data['name']
        self.canvas = data['canvas']
        self.nodes = {}
        self.nets = {}
        
        for i in range(len(data['blocks'])):
            self.nodes[data['blocks'][i]['id']] = {'width': data['blocks'][i]['width'], 'height': data['blocks'][i]['height']}

        for i in range(len(data['nets'])):
            self.nets[data['nets'][i]['net_id']] = set(data['nets'][i]['blocks'])

