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
    
    @staticmethod
    def get_backend_coordinates(backend_name):
        """Returns a list with the qubit coordinates of the specified backend
        """
        ibm_qubit_coords = []
        if backend_name == 'ibm_fez' or backend_name == 'ibm_marrakesh':
            for i in range(8):
                for j in range(16):
                    ibm_qubit_coords.append((2*i, j))
            for i in range(4):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j+3))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+1))
        elif backend_name == 'ibm_torino':
            for i in range(7):
                for j in range(15):
                    ibm_qubit_coords.append((2*i, j))
            for i in range(4):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+2))
        else: # Eagle r3
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
        return sorted(ibm_qubit_coords)
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
            if backend == 'ibm_fez' or backend == "ibm_marrakesh":
                first_qubit = 3
            else:
                first_qubit = 0
        if backend is not None:
            ibm_qubit_coords = HeavyHexLattice.get_backend_coordinates(backend)
        else:
            ibm_qubit_coords = self.coords
        ibm_qubit_coords = sorted(ibm_qubit_coords)
        ibm_origin = ibm_qubit_coords[first_qubit]
        if (ibm_origin[0]+1, ibm_origin[1]) in ibm_qubit_coords:
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
        else:
            raise ValueError("Invalid initial qubit")
    def get_all_layouts(self, backend=None):
        """Returns a list of all possible mappings from own qubit indices to ibm physical qubits
        Possible isomorphisms are reflections and 60º rotations
        Arguments:
            backend:
                Device backend
        """
        # Gets the new rows and columns after a 60º rotation
        # A bit cumbersome due to the brick-wall layout instead of regular hexagons
        def rotated_row(r,c):
            if r != int(r):
                return int(c)//2+1+int(r-0.5)//2
            return [0,0.5,1,1][int(2*r+2*c)%4]+int(r)//2+int(2*(c+r%2))//4
        def rotated_column(r,c):
            if r != int(r):
                return -(int(c)//2)+0.5+3*(int(r-0.5)//2)+2*(int(r-0.5)%2)
            return [0,0,0,-0.5][int(2*r+2*c)%4]+int(3*r)//2-(int(2*(c-r%2))//4)
        layouts = []
        sorted_layouts = []
        ibm_qubit_coords = HeavyHexLattice.get_backend_coordinates(backends_objs_to_names(backend))
        for (row_ibm, col_ibm) in ibm_qubit_coords:
            for dir_col in [1, -1]: # left-right mirroring
                for dir_row in [1, -1]: # up-down mirroring
                    layout = []
                    fits = False
                    if (row_ibm+dir_row, col_ibm) in ibm_qubit_coords:
                        if dir_row == -1: # skip up-down mirroring as it is redundant
                            continue
                        fits = True
                        (row_orig, col_orig) = self.coords[0] # Coordinates of qubit 0
                        for (row_lattice, col_lattice) in self.coords:
                            offset = (int(2*(row_lattice-row_orig)), int(2*(col_lattice-col_orig)))
                            ibm_c = (row_ibm+dir_row*offset[0], col_ibm+dir_col*offset[1])
                            if ibm_c not in ibm_qubit_coords:
                                fits = False
                                break
                            layout.append(ibm_qubit_coords.index(ibm_c))
                    elif (row_ibm+dir_row, col_ibm+2*dir_col) in ibm_qubit_coords: # 60º rotated layouts
                        fits = True
                        (row_orig, col_orig) = self.coords[0] # Coordinates of qubit 0
                        (row_orig, col_orig) = (rotated_row(row_orig, col_orig), rotated_column(row_orig, col_orig))
                        for (row_lattice, col_lattice) in self.coords:
                            (row_lattice, col_lattice) = rotated_row(row_lattice, col_lattice), rotated_column(row_lattice, col_lattice)
                            offset = (int(2*(row_lattice-row_orig)), int(2*(col_lattice-col_orig)))
                            ibm_c = (row_ibm+dir_row*offset[0], col_ibm+dir_col*offset[1])
                            if ibm_c not in ibm_qubit_coords:
                                fits = False
                                break
                            layout.append(ibm_qubit_coords.index(ibm_c))
                    if fits:
                        sort = sorted(layout)
                        # Exclude automorphisms
                        if sort not in sorted_layouts:
                            layouts.append(layout)
                            sorted_layouts.append(sort)
        return layouts

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
        mirror = False
        if number_qubits:
            if backend is None:
                start = 0 if first_qubit is None else first_qubit
                labels = [str(start + i) for i in range(len(self.coords))]
            else:
                backend = backends_objs_to_names(backend)
                initial_layout = self.initial_qubit_layout(first_qubit, backend)
                labels = [str(qb) for qb in initial_layout]
                if initial_layout[1] < initial_layout[0]:
                    mirror = True
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
        if mirror:
            ax.invert_xaxis()
        plt.ylim([-0.2, self.max_y+0.2])
        plt.tight_layout()
        plt.rcdefaults()

def is_edge_coords(coords):
    if type(coords) == tuple:
        return (coords[0] % 1 != 0) or (coords[1] % 1 != 0)
    else:
        try:
            coords = np.array(coords)
            return ~np.equal(coords.sum(axis=1) % 1, 0)
        except np.AxisError:
            raise np.AxisError("Coords is not a valid tuple or N x 2 array")