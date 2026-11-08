import os

os.environ["USE_PATH_FOR_GDAL_PYTHON"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pop_hstm.cli import main_entry


if __name__ == "__main__":
    main_entry()
