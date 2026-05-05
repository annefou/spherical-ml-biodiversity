NOTEBOOKS = [
    "01_flat_cnn_failure_mode",
    "02_rotation_equivariance",
    "03_why_healpix",
    "04_mhw_detection_copernicus_marine",
    "05_real_mhw_and_biodiversity",
]

IMAGES = {
    "01_flat_cnn_failure_mode":          ["images/flat_cnn_failure_mode.png"],
    "02_rotation_equivariance":          ["images/rotation_not_translation.png",
                                          "images/equator_detector_fails_at_pole.png"],
    "03_why_healpix":                    ["images/iso_latitude_rings.png"],
    "04_mhw_detection_copernicus_marine":["images/spherical_ml_payoff.png"],
    "05_real_mhw_and_biodiversity":      ["images/mhw_healpix_western_australia_2011.png",
                                          "images/mhw_biodiversity_overlay_2011.png"],
}


rule all:
    input:
        [img for nb in NOTEBOOKS for img in IMAGES[nb]],


rule run_notebook:
    input:
        script="notebooks/{name}.py",
    output:
        notebook="notebooks/{name}.ipynb",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute --inplace {output.notebook}
        """


# One concrete target per notebook so Snakemake can resolve outputs.
for _nb, _imgs in IMAGES.items():
    rule:
        name:
            f"images_{_nb}"
        input:
            f"notebooks/{_nb}.ipynb",
        output:
            *_imgs
        shell:
            "true  # outputs are produced by run_notebook"
