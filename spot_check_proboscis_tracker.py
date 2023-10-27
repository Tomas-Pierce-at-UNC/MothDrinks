
import json
import cine
from matplotlib import pyplot
import itertools
import random
import tube2
import re

DATE_EXPR = re.compile("\d{4}.\d{2}.\d{2}")
MOTH_EXPR = re.compile("moth\d{2}|mothM\d|mothDelta\d")

def draw_object(ax, entry):
    row = entry[1]
    col = entry[2]
    maxrow = entry[10]
    message = str(round(entry[5],3))
    ax.hlines(row, col - 5, col + 5, color='red')
    ax.vlines(col, row - 5, row + 5, color='red')
    ax.hlines(maxrow, 5, 95, color='green')
    ax.text(col + 6, row, message)


if __name__ == '__main__':
    with open("proboscis_measurements3-net.json") as pp:
        data_net = json.load(pp)

    names = list(data_net.keys())
    good = [name for name in names if not "unsuitable" in name]
    fig, ax = pyplot.subplots()
    for name in good:
        date = re.findall(DATE_EXPR, name)[0]
        moth = re.findall(MOTH_EXPR, name)[0]
        try:
            vid = cine.Cine(name)
            measurements = data_net[name]
            entries = filter(lambda item : item[5] > 0.9, measurements)
            groups = itertools.groupby(entries, key=lambda entry : entry[0])
            gs = list(map(lambda group : max(group[1], key=lambda entry : entry[10]),
                     groups))
            mysample = random.choices(gs, k=10)
            for entry in mysample:
                index = entry[0]
                f = vid.get_ith_image(index)
                frame = tube2.tube_crop1(f)
                ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
                draw_object(ax, entry)
                fig.savefig(f"ptests/p_test-{moth}-{date}-{index}.png")
                fig.clear()
                fig.add_axes(ax)
        finally:
            vid.close()
