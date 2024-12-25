from biotsavart import (
    calc_biotsavart,
    make_field_file_from_coils,
    get_field_on_grid_numba_parallel,
)
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from plot_modes import plot_modes
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
    field_modes = "field_modes.png"
    grid_file_gold_rec = "golden_record/biotsavart.inp"
    current_file_gold_rec = "golden_record/cur_asd.dd"
    coil_file_gold_rec = "golden_record/co_asd.dd"
    field_file_gold_rec = "golden_record/field.dat"
    field_modes_gold_rec = "golden_record/field_modes.png"
    return (
        field_file,
        field_modes,
        grid_file_gold_rec,
        current_file_gold_rec,
        coil_file_gold_rec,
        field_file_gold_rec,
        field_modes_gold_rec,
    )


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


@pytest.fixture
def backup_and_cleanup():
    """First, get the file paths for the golden record input/output files and the output which will be calculated in the test.
    Then backup files which may be overwritten and yield the paths of the files. Afterwards restore the backuped files and delete
    files and folders created during the test.

    Yields:
        field_file (str): Output file of the test Biot-Savart calculation containing the magnetic field values
        field_modes (str): Output file of the Fourier transformation part: Plotted magnetic field modes
        grid_file_gold_rec (str): Original input file defining the grid for the Biot-Savart calculation
        current_file_gold_rec (str): Original input file defining the coil currents for the Biot-Savart calculation
        coil_file_gold_rec (str): Original input file defining the magnetic coils for the Biot-Savart calculation
        field_file_gold_rec (str): Original output file of the Biot-Savart calculation containing the magnetic field values
        field_modes_gold_rec (str): Original output file of the Fourier transformation part: Plotted magnetic field modes
    """
    (
        field_file,
        field_modes,
        grid_file_gold_rec,
        current_file_gold_rec,
        coil_file_gold_rec,
        field_file_gold_rec,
        field_modes_gold_rec,
    ) = get_filenames()
    protected_files = [field_file, field_modes]
    temp_files = backup_files(protected_files)

    yield field_file, field_modes, grid_file_gold_rec, current_file_gold_rec, coil_file_gold_rec, field_file_gold_rec, field_modes_gold_rec

    cleanup_files(protected_files)
    restore_backups(temp_files)


def get_head_of_field_file(fname):
    """Get the head (=first four lines) of the Biot-Savart field calculation output file.

    Args:
        fname (str): Name of the Biot-Savart field calculation output file

    Returns:
        head (array[float], shape=(4,4)): Values of the first four lines in this file
    """
    with open(fname, "r") as f:
        head = [next(f).strip().split(" ") for _ in range(4)]
        head = [np.pad(row, (0, 4 - len(row))) for row in head]
        head = np.array(head).astype(float)

    return head


def merge_images(img_name1, img_name2, title1, title2):
    """Merge two images vertically and add titles to them.

    Args:
        img_name1 (str): Name of the upper image
        img_name2 (str): Name of the lower image
        title1 (str): Title of the first image
        title2 (str): Title of the second image

    Returns:
        Image (PIL.Image.Image): Merged image
    """
    img1 = Image.open(img_name1)
    img2 = Image.open(img_name2)

    both_width = max(img1.width, img2.width)
    both_height = img1.height + img2.height
    both_img = Image.new("RGB", (both_width, int(both_height * 1.1)), color="black")

    both_img.paste(img1, (0, int(both_height * 0.05)))
    both_img.paste(img2, (0, int(img1.height + both_height * 0.1)))

    both_draw = ImageDraw.Draw(both_img)
    font = ImageFont.truetype("arial.ttf", 20)
    _, _, w1, h1 = both_draw.textbbox((0, 0), title1, font=font)
    both_draw.text(
        (img1.width / 2 - w1 / 2, both_height * 0.025 - h1 / 2),
        title1,
        fill="white",
        font=font,
    )
    _, _, w2, h2 = both_draw.textbbox((0, 0), title2, font=font)
    both_draw.text(
        (img2.width / 2 - w2 / 2, img1.height + both_height * 0.075 - h2 / 2),
        title2,
        fill="white",
        font=font,
    )
    return both_img


def test_field_against_golden_record(backup_and_cleanup):
    (
        field_file,
        field_modes,
        grid_file_gold_rec,
        current_file_gold_rec,
        coil_file_gold_rec,
        field_file_gold_rec,
        field_modes_gold_rec,
    ) = backup_and_cleanup

    # The field periodicity from the golden record is the last number in the first line of the output file.
    with open(field_file_gold_rec, "r") as f:
        field_periodicity = int(f.readline().strip().split(" ")[-1])

    # Create a new field_file using the golden record inputs.
    make_field_file_from_coils(
        grid_file_gold_rec,
        coil_file_gold_rec,
        current_file_gold_rec,
        field_file,
        calc_biotsavart,
        get_field_on_grid_numba_parallel,
        field_periodicity,
    )

    # Check the head (first 4 lines) of the fresh output and the golden record.
    head_new = get_head_of_field_file(field_file)
    head_gold_rec = get_head_of_field_file(field_file_gold_rec)
    assert head_new.shape == head_gold_rec.shape, "Shape mismatch of the output head"
    assert np.allclose(
        head_new, head_gold_rec, atol=1e-10
    ), "Head values do not match golden record"

    # Check the magnetic field values of the fresh output and the golden record
    B_new = np.loadtxt(field_file, skiprows=4)
    B_gold_rec = np.loadtxt(field_file_gold_rec, skiprows=4)
    assert B_new.shape == B_gold_rec.shape, "Shape mismatch of field files"
    assert np.allclose(
        B_new, B_gold_rec, rtol=1e-5, atol=1e-5
    ), "Magnetic field values do not match golden record"

    # Check by eye the output of the Fourier transformation of the
    # magnetic field from plot_modes.py against the golden record:

    plot_modes(field_file, n_modes=8)

    merged_img = merge_images(
        field_modes_gold_rec, field_modes, "Golden Record", "New Calculation"
    )
    merged_img.show()
