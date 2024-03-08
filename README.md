# Achieving sub-0.5-angstrom–resolution ptychography in an uncorrected electron microscope

This repo presents my personal collection of code and data used for the paper [Achieving sub-0.5-angstrom–resolution ptychography in an uncorrected electron microscope](https://www.science.org/doi/10.1126/science.adl2029) published in Science (2024).

Our work demonstrates that with electron ptychography, one may achieve deep sub-angstron resolution on an uncorrected electron microscope.

While the [zenodo record](https://zenodo.org/doi/10.5281/zenodo.7964153) is meant for permanently archiving the raw data and relevant codes, I'll try to maintain this personal repo as much as possible.

## Environment and dependencies

- You'll (probably) need Matlab version >= 2019 to run `fold_slice`. I've only tested up to 2023b on Windows and 2021a on Linux
- You'll need a CUDA-compatible GPU with ideally 8 or 16 GB of RAMs that is detectable by Matlab
Note: If you don't have a Matlab license, it is possible to run the reconstructions with other packages like [`py4DSTEM`](https://github.com/py4dstem/py4DSTEM) if you modify certain inputs like the CBED orientation, scan directions, etc.

## Usage

If you download this folder through my [github](https://github.com/chiahao3), the sub-folders under `/data` should be empty due to the file size limit on Github.
You would need to download the actual experimental data from our [zenodo record](https://zenodo.org/doi/10.5281/zenodo.7964153) 
Put the downloaded `scan_x128_y128.raw` into the corresponding folders and then you should be able to run reconstruction scripts from the main directory `uncorrected_ptycho/`.

For example:
1. Download and unzip `Fig_01.zip` from our [zenodo record](https://zenodo.org/doi/10.5281/zenodo.7964153) 
2. Copy the `Fig_01/Panel_c-d_Talos/scan_x128_y128.raw`, and paste it to `data/Fig_1d_10.3mrad_Talos/`
3. Run reconstruction scripts `Fig_1d_10p3mrad_Talos_runPtycho.m`
4. The ptychographic reconstruciton output would be saved to `data/Fig_1d_10.3mrad_Talos/`

### Additional notes

- The first time executing `fold_slice` might takes longer because Matlab would need to compile the MEX functions for GPU
- The script would first read the 4D-STEM data (`.raw`), do some preprocessing, and save it as .hdf5 for later reconstruction. Saving an upsampled .hdf5 (a few GB) would take a couple minutes.
- Practically we only need to preprocess the `.raw` once and you may reuse the `.hdf5` if you're reconstrucing the same data multiple times, but you'll need to modify the provided script accordingly because this script is meant for out-of-the-box experience.
- Generally, a successful reconstruciton would need a couple stages of refinement to achive the best results, while these example scripts are meant for demo purpose so the result could be sub-optimal due to insufficient iterations / probe modes.
- If you run into GPU memory issues, try reducing the `grouping`, `resample_factor`, or `Nprobe`.
- There're a lot more useful information provided by Yi that's been compiled in [`fold_slice`](https://github.com/yijiang1/fold_slice). I highly recommend cheking out the [tutorials](https://anl.app.box.com/s/f7lk410lf62rnia70fztd5l7n567btyv).


## References

If you find the code useful for your research, please consider citing our original paper
```bib
@article{nguyen2024achieving,
  title={Achieving sub-0.5-angstrom--resolution ptychography in an uncorrected electron microscope},
  author={Nguyen, Kayla X and Jiang, Yi and Lee, Chia-Hao and Kharel, Priti and Zhang, Yue and van der Zande, Arend M and Huang, Pinshane Y},
  journal={Science},
  volume={383},
  number={6685},
  pages={865--870},
  year={2024},
  publisher={American Association for the Advancement of Science}
}
```

-------------------
Created by Chia-Hao Lee on 2024.03.07

cl2696@cornell.edu