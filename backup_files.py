import os


def backup_files(files):
    """Backup files which may be overwritten by the test by moving them into a temporary folder.

    Args:
        files (iterable): Paths to files to backup in order to protect them from being overwritten

    Returns:
        new_files (iterable): New paths of the files which have been moved to temporary
    """
    if not os.path.exists("temporary"):
        os.mkdir("temporary")
    new_files = files.copy()
    for i, file in enumerate(files):
        if os.path.exists(file):
            os.rename(file, f"temporary/{file}")
            new_files[i] = f"temporary/{file}"
    return new_files


def cleanup_files(files):
    """(CAUTION) Delete all files in input argument list.

    Args:
        files (iterable): Paths to files which are deleted.
    """
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def restore_backups(files):
    """Move files in files outside the temporary folder and delete the folder afterwards.

    Args:
        files (iterable): Files to be moved
    """
    for file in files:
        if os.path.exists(file):
            os.rename(file, file.split("temporary/")[1])
    os.rmdir("temporary")
