from biotsavart_modes.biotsavart.biotsavart import (
    calc_biotsavart,
    make_field_file_from_coils,
    get_field_on_grid_numba_parallel,
)
from biotsavart_modes.plotting.plot_modes import (
    plot_modes,
    read_field,
    read_field_hdf5,
    read_field_netcdf,
)
from tests.helpers.backup_files import backup_files, cleanup_files, restore_backups
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pytest


def get_filenames():
    """Return the paths of the file for the new calculation and the golden record input and output files.

    Returns:
        field_file (WindowsPath): Output field file for the test Biot-Savart calculation
        field_modes (WindowsPath): Output file of the Fourier transformation part: Plotted magnetic field modes
        grid_file_gold_rec (WindowsPath): Original input file defining the grid for the Biot-Savart calculation
        current_file_gold_rec (WindowsPath): Original input file defining the coil currents for the Biot-Savart calculation
        coil_file_gold_rec (WindowsPath): Original input file defining the magnetic coils for the Biot-Savart calculation
        field_file_gold_rec (WindowsPath): Original output file of the Biot-Savart calculation containing the magnetic field values.
        field_modes_gold_rec (WindowsPath): Original output file of the Fourier transformation part: Plotted magnetic field modes.
    """
    field_file = Path("tests/field.h5")
    field_modes = Path("tests/field_modes.png")
    grid_file_gold_rec = Path("tests/golden_record/biotsavart.inp")
    current_file_gold_rec = Path("tests/golden_record/cur_asd.dd")
    coil_file_gold_rec = Path("tests/golden_record/co_asd.dd")
    field_file_gold_rec = Path("tests/golden_record/field.dat")
    field_modes_gold_rec = Path("tests/golden_record/field_modes.png")
    return (
        field_file,
        field_modes,
        grid_file_gold_rec,
        current_file_gold_rec,
        coil_file_gold_rec,
        field_file_gold_rec,
        field_modes_gold_rec,
    )


@pytest.fixture
def backup_and_cleanup():
    """First, get the file paths for the golden record input/output files and the output which will be calculated in the test.
    Then backup files which may be overwritten and yield the new paths of the files. Afterwards restore the backuped files and
    delete files and folders created during the test.

    Yields:
        field_file (WindowsPath): Output file of the test Biot-Savart calculation containing the magnetic field values
        field_modes (WindowsPath): Output file of the Fourier transformation part: Plotted magnetic field modes
        grid_file_gold_rec (WindowsPath): Original input file defining the grid for the Biot-Savart calculation
        current_file_gold_rec (WindowsPath): Original input file defining the coil currents for the Biot-Savart calculation
        coil_file_gold_rec (WindowsPath): Original input file defining the magnetic coils for the Biot-Savart calculation
        field_file_gold_rec (WindowsPath): Original output file of the Biot-Savart calculation containing the magnetic field values
        field_modes_gold_rec (WindowsPath): Original output file of the Fourier transformation part: Plotted magnetic field modes
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


def get_head_of_field_file(field_file):
    """Get the head (=first four lines) of the Biot-Savart field calculation output file.

    Args:
        field_file (WindowsPath): File of the magnetic field calculation output

    Returns:
        head (array[float], shape=(4,4)): Values of the first four lines in this file
    """
    with open(field_file, "r") as f:
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

    grid_gold_rec, BR_gold_rec, Bphi_gold_rec, BZ_gold_rec = read_field(
        field_file_gold_rec
    )
    field_fname = field_file
    if field_fname.endswith(".h5") or field_fname.endswith(".hdf5"):
        grid, BR, Bphi, BZ = read_field_hdf5(field_file)
    elif field_fname.endswith(".nc") or field_fname.endswith(".cdf"):
        grid, BR, Bphi, BZ = read_field_netcdf(field_file)
    else:
        grid, BR, Bphi, BZ = read_field(field_file)

    # Check the grid of the fresh output and the golden record.
    attribute_names = [attr for attr in dir(grid) if not attr.startswith("_")]
    for attr in attribute_names:
        getattr(grid, attr)
        assert np.allclose(
            getattr(grid, attr), getattr(grid_gold_rec, attr), rtol=1e-10
        ), f"Grid value '{attr}' do not match golden record"

    # Check the magnetic field values of the fresh output and the golden record
    assert np.allclose(
        BR, BR_gold_rec, rtol=1e-10
    ), "Radial magnetic field components do not match golden record"
    assert np.allclose(
        Bphi, Bphi_gold_rec, rtol=1e-10
    ), "Azimuthal magnetic field components do not match golden record"
    assert np.allclose(
        BZ, BZ_gold_rec, rtol=1e-10
    ), "Axial magnetic field components do not match golden record"

    # Check by eye the output of the Fourier transformation of the
    # magnetic field from plot_modes.py against the golden record:
    plot_modes(field_file, n_modes=8)

    merged_img = merge_images(
        field_modes_gold_rec, field_modes, "Golden Record", "New Calculation"
    )
    merged_img.show()
