import h5py


def strip_dot(ext):
    return ext.lstrip(".")


def create_ext_glob(ext):
    return "*.{}".format(strip_dot(ext))


def load_hdf5(input_file):
    with h5py.File(input_file, "r") as f:
        return f["image"][()]


def write_hdf5(image, output_file):
    with h5py.File(output_file, "w") as f:
        f.create_dataset("image", data=image, dtype=image.dtype)
