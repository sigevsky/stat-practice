import pandas as pd
from shared import *
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

student_raw_data = [pd.read_excel('data/r_data.xlsx', str(i)) for i in range(1, 29)]


def to_array(path):
    color_image = Image.open(path)
    bw = color_image.convert('1', dither=Image.NONE)
    a = np.array(bw, dtype=int)
    p_path, tail = os.path.split(path)
    np.save(f"{p_path}/{os.path.splitext(tail)[0]}", 1 - a)


def draw_and_save_to_png(path, x, y):
    fig: plt.Figure = plt.figure()
    ax = fig.gca()
    ax.axis('off')
    ax.plot(x, y)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    to_array(path)


def extract_data_from_sheet(raw: pd.DataFrame):
    state_span = 2 * len(TESTS)
    return raw.iloc[2:, 1:state_span + 1]
    # return (raw.iloc[2:, 1:], []) if len(raw.columns) == len(TESTS * 2) + 1 else (raw.iloc[2:, 2:state_span + 3], raw.iloc[2:, state_span + 3: state_span * 2 + 3])


def dump_student_df(prefix, data: pd.DataFrame):
    assert len(data.columns) == len(TESTS * 2)
    for i, test in zip(range(0, len(TESTS) * 2, 2), TESTS):
        x, y = data.iloc[:, i].dropna(), data.iloc[:, i + 1].dropna()
        x_numeric, y_numeric = x[x.map(np.isreal)], y[y.map(np.isreal)]
        draw_and_save_to_png(f"{prefix}/{test}.png", x_numeric, y_numeric)


def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


for i, st_df in enumerate(student_raw_data):
    open_e = extract_data_from_sheet(st_df)
    path = f"raw/{E_OPEN}/{i + 1}"
    safe_mkdir(path)
    dump_student_df(path, open_e)

print("Hello world")

