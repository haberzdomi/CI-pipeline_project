from biotsavart import calc_biotsavart, make_field_file_from_coils
import numpy as np
import os
import pytest


def get_filenames():
    """Return the paths of the file for the new calculation and the golden record input and output files.

    Returns:
        field_file (str): Output file for the test Biot-Savart calculation
        grid_file_gold_rec (str): Original input file defining the grid for the Biot-Savart calculation
        current_file_gold_rec (str): Original input file defining the coil currents for the Biot-Savart calculation
        coil_file_gold_rec (str): Original input file defining the magnetic coils for the Biot-Savart calculation
        field_file_gold_rec (str): Original output file of the Biot-Savart calculation containing the magnetic field values.
    """
    field_file = "field_file"
    grid_file_gold_rec = "golden_record/biotsavart.inp"
    current_file_gold_rec = "golden_record/cur_asd.dd"
    coil_file_gold_rec = "golden_record/co_asd.dd"
    field_file_gold_rec = "golden_record/field.dat"
    return (
        field_file,
        grid_file_gold_rec,
        current_file_gold_rec,
        coil_file_gold_rec,
        field_file_gold_rec,
    )


def backup_files(files):
    """Backup files which may be overwritten by the test by moving them into a temporary folder.

    Args:
        files (iterable): Paths to files to backup in order to protect them from being overwritten

    Returns:
        files (iterable): New paths of the files which have been moved to temporary
    """
    if not os.path.exists("temporary"):
        os.mkdir("temporary")
    for i, file in enumerate(files):
        if os.path.exists(file):
            os.rename(file, f"temporary/{file}")
            files[i] = f"temporary/{file}"
    return files


def cleanup_files(files):
    """(CAUTION) Delete all files in files.

    Args:
        files (iterable): Paths to files which are removed.
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
            os.rename(f"./temporary/{file}", file)
    os.rmdir("./temporary")


@pytest.fixture
def backup_and_cleanup():
    """First, get the file paths for the golden record input/output files and the output which will be calculated in the test.
    Then backup files which may be overwritten and yield the paths of the files. Afterwards restore the backuped files and delete
    files and folders created during the test.

    Yields:
        field_file (str): Output file for the test Biot-Savart calculation
        grid_file_gold_rec (str): Original input file defining the grid for the Biot-Savart calculation
        current_file_gold_rec (str): Original input file defining the coil currents for the Biot-Savart calculation
        coil_file_gold_rec (str): Original input file defining the magnetic coils for the Biot-Savart calculation
        field_file_gold_rec (str): Original output file of the Biot-Savart calculation containing the magnetic field values.

    """
    (
        field_file,
        grid_file_gold_rec,
        current_file_gold_rec,
        coil_file_gold_rec,
        field_file_gold_rec,
    ) = get_filenames()
    # field_file_temp = backup_files([field_file])

    yield field_file, grid_file_gold_rec, current_file_gold_rec, coil_file_gold_rec, field_file_gold_rec

    # cleanup_files([field_file])
    # restore_backups([field_file_temp])


def test_field_against_golden_record(backup_and_cleanup):

    (
        field_file,
        grid_file_gold_rec,
        current_file_gold_rec,
        coil_file_gold_rec,
        field_file_gold_rec,
    ) = backup_and_cleanup

    # The field periodicity from the golden record is the last number in the first line of the output file.
    with open(field_file_gold_rec, "r") as f:
        field_periodicity = float(f.readline().strip().split(" ")[-1])

    # Create a new field_file using the golden record inputs
    make_field_file_from_coils(
        grid_file_gold_rec,
        coil_file_gold_rec,
        current_file_gold_rec,
        field_file,
        calc_biotsavart,
        field_periodicity,
    )

    # Get the head (first 4 lines) of the fresh output and the golden record.
    with open(field_file, "r") as f:
        head_new = [next(f).strip().split(" ") for _ in range(4)]
        head_new = [np.pad(row, (0, 4 - len(row))) for row in head_new]
        head_new = np.array(head_new).astype(float)
    with open(field_file_gold_rec, "r") as f:
        head_gold_rec = [next(f).strip().split(" ") for _ in range(4)]
        head_gold_rec = [np.pad(row, (0, 4 - len(row))) for row in head_gold_rec]
        head_gold_rec = np.array(head_gold_rec).astype(float)

    # Get the magnetic field values of the fresh output and the golden record
    B_new = np.loadtxt(field_file, skiprows=4)
    B_gold_rec = np.loadtxt(field_file_gold_rec, skiprows=4)

    assert head_new.shape == head_gold_rec.shape, "Shape mismatch of the output head"
    assert np.allclose(
        head_new, head_gold_rec, atol=1e-10
    ), "Head values do not match golden record"

    assert B_new.shape == B_gold_rec.shape, "Shape mismatch of field files"
    assert np.allclose(
        B_new, B_gold_rec, rtol=1e-5, atol=1e-5
    ), "Magnetic field values do not match golden record"
