import os

def make_dir(dir_name):
    """Create directory if it does not exist

    Args:
        dir_name (str): directory name

    Returns:
        str: directory name
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    directory=dir_name
    return directory