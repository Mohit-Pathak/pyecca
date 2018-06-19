from IPython.display import Image
from casadi.tools.graph import graph
import casadi as ca


def draw_graph(x):
    g = graph.dotgraph(x)
    # g.set('dpi', 300)
    png = g.create('dot', 'png')
    return Image(png)