# Contributor: César Benito Lamata @cesarBLG

from utils.hexec import backends_objs_to_names
from functools import cached_property
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --------------------------------------
#     HEXAGONAL LATTICE ABSTRACTIONS
# --------------------------------------

# Stores qubit index and coordinates for every vertex (matter qubit)
# Two types of nodes: downwards and upwards (for the schedule)
class vertex:
    def __init__(self, coords, qubit):
        self.coords = coords
        self.qubit = qubit
        self.downwards = (coords[0]+coords[1])%2 == 1

    def __repr__(self):
        return f"<Vertex {self.coords} -> Qb: {self.qubit}>"

# Stores qubit index and coordinates for every edge (gauge qubit)
# Type indicates link direction (not used)
# Color represents a three-colouring of all links (not used)
class edge:
    def __init__(self, coords, qubit):
        self.coords = coords
        self.qubit = qubit
        if int(coords[0]) == coords[0]:
            self.color = [1,0,2][int(coords[1]-0.5)%3]
            self.type = ['x','y'][int(coords[0]+coords[1]-0.5)%2]
        else:
            self.color = [0,1,2][(coords[1]//2+int(coords[0]-0.5)%2+2)%3]
            self.type = 'z'

    def __repr__(self):
        return f"<Edge {self.coords} -> Qb: {self.qubit}>"

# Stores coordinates of all qubits in the heavy-hex lattice
# Additionally, stores edges and vertices in separate lists
class HeavyHexLattice:
    def __init__(self, plaq_width, plaq_height):
        self.plaquettes_width = plaq_width
        self.plaquettes_height = plaq_height
        width=plaq_width*2+1
        height=plaq_height
        edge_coords = []
        vertex_coords = []
        for i in range(height+1):
            for j in range(width+1):
                if i==0 and j == 0:
                    continue
                if i == height:
                    if j == 0 and i%2 != 0:
                        continue
                    if j == width and i%2 == 0:
                        continue
                vertex_coords.append((i,j))
        for i in range(height+1):
            for j in range(width):
                if i==0 and j == 0:
                    continue
                if i == height:
                    if j == 0 and i%2 != 0:
                        continue
                    if j == width-1 and i%2 == 0:
                        continue
                edge_coords.append((i, j+0.5))
        for i in range(height):
            if i%2 == 0:
                for j in range((width+1)//2):
                    edge_coords.append((i+0.5, 2*j+1))
            else:
                for j in range(width//2+1):
                    edge_coords.append((i+0.5, 2*j))
        self.coords = sorted(edge_coords+vertex_coords)
        self.edges = dict()
        self.vertices = dict()
        for (q,c) in enumerate(self.coords):
            if c in edge_coords:
                self.edges[c] = edge(c, q)
            else:
                self.vertices[c] = vertex(c, q)

    @cached_property
    def node_coords(self):
        return list(self.vertices.keys())

    @cached_property
    def edge_coords(self):
        return list(self.edges.keys())

    def coords_to_logical_qb(self, coords):
        if len(coords) == 2:
            coords = [tuple(coords)]
        return [self.coords.index((coord[0], coord[1])) for coord in coords]
    
    def edges_connected_to_node(self, node_coords):
        connected_edges_coords = []
        for coords in self.coords:
            if (0 <= (dx := abs(node_coords[0] - coords[0])) < 1) and (0 <= (dy := abs(node_coords[1] - coords[1])) < 1):
                if not (dx == 0 and dy == 0):
                    connected_edges_coords.append(coords)
        return connected_edges_coords

    def initial_qubit_layout(self, first_qubit=None, backend=None, reflex=None):
        """Returns the mapping from own qubit indices to ibm physical qubits
        Arguments:
            first_qubit:
                The index of the top-left physical qubit to be used
                Default value: the top-left qubit of the top-left plaquette in the device
            backend:
                Device backend
            reflex:
                Whether to mirror the layout from left to right
                Default: only mirror when the layout does not fit if unmirrored
        """
        backend = backends_objs_to_names(backend)
        ibm_qubit_coords = []
        if first_qubit == None:
            if backend == 'ibm_fez':
                first_qubit = 3
            else:
                first_qubit = 0
        if backend == 'ibm_fez':
            for i in range(8):
                for j in range(16):
                    ibm_qubit_coords.append((2*i, j))
            for i in range(4):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j+3))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+1))
        elif backend == 'ibm_torino':
            for i in range(7):
                for j in range(15):
                    ibm_qubit_coords.append((2*i, j))
            for i in range(4):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+2))
        elif backend != None: # Eagle r3
            for i in range(7):
                for j in range(15):
                    if i == 0 and j == 14:
                        continue
                    if i == 6 and j == 0:
                        continue
                    ibm_qubit_coords.append((2*i, j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+2))
        else:
            ibm_qubit_coords = self.coords
        ibm_qubit_coords = sorted(ibm_qubit_coords)
        ibm_origin = ibm_qubit_coords[first_qubit]
        if (ibm_origin[0]+1, ibm_origin[1]) not in ibm_qubit_coords:
            raise ValueError("Invalid initial qubit")
        if reflex is None:
            if (ibm_origin[0]+2,ibm_origin[1]-2) not in ibm_qubit_coords:
                reflex = True
            else:
                reflex = False
        if reflex:
            own_origin = self.coords[0]
            for c in self.coords:
                if c[0] == 0 and c[1] > own_origin[1]:
                    own_origin = c
        else:
            own_origin = self.coords[0]
        initial_qubit_layout = []
        for c in self.coords:
            offset = (int(2*(c[0]-own_origin[0])), int(2*(c[1]-own_origin[1])))
            if reflex:
                offset = (offset[0],-offset[1])
            ibm_c = (ibm_origin[0]+offset[0],ibm_origin[1]+offset[1])
            initial_qubit_layout.append(ibm_qubit_coords.index(ibm_c))
        return np.array(initial_qubit_layout)

    def __len__(self):
        return len(self.coords)

    @cached_property
    def max_x(self):
        return max([coords[1] for coords in self.coords])
    
    @cached_property
    def max_y(self):
        return max([coords[0] for coords in self.coords])
    
    def plot_lattice(self, scale=1.5, number_qubits=False, first_qubit=None, backend=None):
        plt.rc("font", family="serif")
        vertex_x = np.array([v[1] for v in sorted(list(self.vertices.keys()))], dtype=int)
        vertex_y = np.array([self.max_y - v[0] for v in sorted(list(self.vertices.keys()))], dtype=int)
        edges_endpoints_x = np.array([[np.floor(e[1]), np.ceil(e[1])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_endpoints_y = np.array([[self.max_y - np.floor(e[0]), self.max_y - np.ceil(e[0])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_boxes_x = np.mean(edges_endpoints_x, axis=1)
        edges_boxes_y = np.mean(edges_endpoints_y, axis=1)
        
        fig, ax = plt.subplots(figsize=[scale*self.max_x, scale*self.max_y])
        ax.set_aspect('equal')
        for exs, eys in zip(edges_endpoints_x, edges_endpoints_y):
            plt.plot(exs, eys, color="black")
        plt.scatter(vertex_x, vertex_y, 350*scale, marker="o", c="white", edgecolors="black", zorder=2)
        plt.scatter(edges_boxes_x, edges_boxes_y, 400*scale, marker=(4, 0, 45), c="white", edgecolors="black", zorder=2)
        if number_qubits:
            if backend is None:
                labels = [str(first_qubit + i) for i in range(len(self.coords))]
            else:
                backend = backends_objs_to_names(backend)
                labels = [str(qb) for qb in self.initial_qubit_layout(first_qubit, backend)]
            ttransform = mpl.transforms.Affine2D().translate(0, -1*scale)
            for i, (y, x) in enumerate(self.coords):
                text = plt.text(x, self.max_y - y, labels[i], horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale})
                text.set_transform(text.get_transform() + ttransform)
        else:
            for i, (x, y) in enumerate(zip(vertex_x, vertex_y)):
                plt.text(x, y, r"$\tau$", horizontalalignment="center", verticalalignment="center", fontdict={"size": 10*scale, "family":"serif"})
            for i, (x, y) in enumerate(zip(edges_boxes_x, edges_boxes_y)):
                plt.text(x, y, r"$\sigma$", horizontalalignment="center", verticalalignment="center", fontdict={"size": 10*scale, "family":"serif"})
        plt.axis("off")
        if backend is not None:
            if backend != 'ibm_fez' and backend != None:
                ax.invert_xaxis()
        plt.tight_layout()
        plt.rcdefaults()
