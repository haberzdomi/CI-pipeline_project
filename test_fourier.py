from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import pytest
from reference_solutions import fourier_analysis


@pytest.mark.parametrize(
    "n_max, field_file",
    [
        [
            8,
            "field.h5",
        ],
    ],
)
def test_fourier(n_max, field_file):
    """Test the handwritten discrete fourier transform of the magnetic field
    components against the solution from numpy.fft.fft. Both calculation
    methods are only allowed to deviate in small numerical errors. For the
    test, the real and imaginary parts of the first 'n_max' modes are
    calculated on a grid in radial dimension defined by R_min, R_max and nR.
    The z-component is fixed to the middle value of the grid defined by Z_min,
    Z_max and nZ, i.e on the index nZ//2. These input parameters are read from
    field_file. The solutions are also compared visually in a line plot for
    each of the first 'n_max' modes.

    Args:
        n_max (int): highest number of magnetic field mode up to which the test is done.
        field_file (str): File name of the magnetic field calculation output.
    """

    # Calculate the magnetic field components for the first 'n_max' modes
    # for the radial grid R via handwritten fourier transform and via numpy.fft.ftt
    R, BnR, BnR_fft = fourier_analysis(n_max, field_file)

    # Create 4 subplots in a row before starting the next row.
    n_rows = n_max // 4  # Number of rows for subplots
    # Minumum and maximum limit for scientific notation of axes labels
    rcParams["axes.formatter.limits"] = (-3, 4)
    fig = plt.figure(figsize=(12, 6))
    axs = fig.subplots(n_rows, 4).ravel()
    for k in range(n_max):
        # For each magnetic field mode 'k' plot the real and imaginary part for both calculation variants.
        axs[k].plot(R, BnR[k].real, linestyle="-", label="real part, handwritten calc.")
        axs[k].plot(
            R,
            BnR_fft[k].real,
            linestyle="--",
            dashes=(3, 5),
            label="real part, FFT calc.",
        )
        axs[k].plot(R, BnR[k].imag, linestyle="-", label="imag part, handwritten calc.")
        axs[k].plot(
            R,
            BnR_fft[k].imag,
            linestyle="--",
            dashes=(3, 5),
            label="imag part, FFT calc.",
        )

        # Axis labeling and title
        axs[k].set_xlabel("R")
        axs[k].set_ylabel("B_n(R)")
        axs[k].set_title(f"$n = {k + 1}$")

    # Get one legend for all subplots at the top.
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.01, 0.94), loc="lower left", ncol=4)

    # Adjust subplot positions
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    #plt.show()

    # Only small floating point deviations of the two calculation methods are allowed.
    rel_errors_real = np.abs(BnR.real - BnR_fft.real) / np.abs(BnR_fft.real)
    assert_msg_real = "The two calculation methods lead to different solutions for the real parts of the magnetic field modes"
    rel_errors_imag = np.abs(BnR.imag - BnR_fft.imag) / np.abs(BnR_fft.imag)
    assert_msg_imag = "The two calculation methods lead to different solutions for the imaginary parts of the magnetic field modes"

    assert_msg_both = "The two calculation methods lead to different solutions for the real and imaginary parts of the magnetic field modes"

    assert np.all(rel_errors_real < 1e-7) and np.all(
        rel_errors_imag < 1e-7
    ), assert_msg_both
    assert np.all(rel_errors_real < 1e-7), assert_msg_real
    assert np.all(rel_errors_imag < 1e-7), assert_msg_imag
