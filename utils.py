import json
def save_json(object, file_path):
    with open(file_path, mode='w') as f:
        json.dump(object, f, indent=4)


def open_json(file_path):
    with open(file_path, mode='r') as f:
        json_object = json.load(f)
        return json_object
