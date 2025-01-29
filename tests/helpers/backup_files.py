import os
from importlib.resources import files


def backup_files(files):
    """Backup files which may be overwritten by the test by moving them into a temporary folder.

    Args:
        files (iterable): Paths to files to backup in order to protect them from being overwritten

    Returns:
        moved_files (iterable): New paths of the files which have been moved to temporary
    """

    moved_files = files.copy()
    for i, file in enumerate(files):
        if os.path.exists(file):
            temp_folder = "temporary"
            if not os.path.exists(temp_folder):
                os.mkdir(temp_folder)
            moved_file = f"temporary/{file}"
            os.rename(file, moved_file)
            moved_files[i] = moved_file
        
    return moved_files


def cleanup_files(files):
    """(CAUTION) Delete all files in input argument list.

    Args:
        files (iterable): Paths to files which are deleted.
    """
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            if not os.listdir("temporary"):
                os.rmdir("temporary")


def restore_backups(files):
    """Move files in files outside the temporary folder and delete the folder afterwards.

    Args:
        files (iterable): Files to be moved
    """
    for file in files:
        if os.path.exists(file):
            os.rename(file, str(file).replace("temporary\\", ""))
            if not os.listdir("temporary"):
                os.rmdir("temporary")
