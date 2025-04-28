# Contributor: César Benito Lamata @cesarBLG

from utils.hexec import backends_objs_to_names
from utils.plotting import best_text_color
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
    def node_qbinds(self):
        return [i for i, coords in enumerate(self.coords) if coords in self.node_coords]

    @cached_property
    def edge_coords(self):
        return list(self.edges.keys())
    
    @cached_property
    def edge_qbinds(self):
        return [i for i, coords in enumerate(self.coords) if coords in self.edge_coords]

    def coords_to_logical_qb(self, coords):
        if isinstance(coords, tuple):
            coords = [tuple(coords)]
        to_return = [self.coords.index((coord[0], coord[1])) for coord in coords]
        return to_return if len(to_return) > 1 else to_return[0]
    
    def edges_connected_to_node(self, node_coords):
        connected_edges_coords = []
        for coords in self.coords:
            if (0 <= (dx := abs(node_coords[0] - coords[0])) < 1) and (0 <= (dy := abs(node_coords[1] - coords[1])) < 1):
                if not (dx == 0 and dy == 0):
                    connected_edges_coords.append(coords)
        return connected_edges_coords
    
    def nodes_connected_to_edge(self, edge_coords):
        return (np.floor(edge_coords[0]).astype(int), np.floor(edge_coords[1]).astype(int)), (np.ceil(edge_coords[0]).astype(int), np.ceil(edge_coords[1]).astype(int))
    
    @staticmethod
    def get_backend_coordinates(backend):
        """Returns a list with the qubit coordinates of the specified backend
        """
        backend_name = backends_objs_to_names(backend)
        ibm_qubit_coords = []
        if backend_name in ["ibm_fez", "ibm_marrakesh", "ibm_aachen", "ibm_kingston"]:
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

    def get_all_layouts(self, backend=None, exclude_automorphisms=False):
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
                        if exclude_automorphisms and dir_row == -1: # skip up-down mirroring as it is an automorphism
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
                        if exclude_automorphisms:
                            sort = sorted(layout)
                            if sort not in sorted_layouts:
                                layouts.append(layout)
                                sorted_layouts.append(sort)
                        elif layout not in layouts:
                            layouts.append(layout)
        return layouts
    
    def line_of_qubits(self, initial_qubit, dvaxis_angle):
        """Returns the list of qubits corresponding to a line in the lattice.
           The line is defined by an initial qubit and a direction parameter than can take these values:
            "h": Horizontal
            0: Vertical
            n > 0: Horizontal angle in terms of qubits advanced in the horizontal direction per plaquette"""
        try:
            initial_qubit_coords = self.coords[initial_qubit]
        except IndexError:
            ValueError("Initial qubit is not in the lattice")
        if dvaxis_angle == "h":
            def new_coords_func(coords):
                candidate = (coords[0], coords[1]+1)
                if candidate in self.coords:
                    return candidate
                else:
                    return None
        elif dvaxis_angle == 0:
            def new_coords_func(coords):
                candidate = (coords[0]+1, coords[1])
                if candidate in self.coords:
                    return candidate
                else:
                    return None
        else:
            def new_coords_func(coords):
                final_candidate = (coords[0]+1, coords[1]+dvaxis_angle)
                middle_candidate = (coords[0]+0.5, coords[1]+dvaxis_angle/2)
                to_return = []
                if middle_candidate in self.coords:
                    to_return.append(middle_candidate)
                if final_candidate in self.coords:
                    to_return.append(final_candidate)
                if len(to_return) < 2:
                    if len(to_return) == 1:
                        return to_return[0]
                    else:
                        return None
                else:
                    return to_return
                
        line_coords = [initial_qubit_coords]
        next_coord = new_coords_func(line_coords[-1])
        while next_coord is not None:
            if isinstance(next_coord, tuple):
                line_coords.append(next_coord)
            else:
                line_coords += next_coord
            next_coord = new_coords_func(line_coords[-1])
        line_qbinds = self.coords_to_logical_qb(line_coords)
        return line_qbinds

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

    def plot_line_of_qubits(self, line_of_qubits, scale=1.5, number_qubits=False, join_line=True, color=None, filepath=""):
        plt.rc("font", family="serif")
        vertex_x = np.array([v[1] for v in sorted(list(self.vertices.keys()))], dtype=int)
        vertex_y = np.array([self.max_y - v[0] for v in sorted(list(self.vertices.keys()))], dtype=int)
        edges_endpoints_x = np.array([[np.floor(e[1]), np.ceil(e[1])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_endpoints_y = np.array([[self.max_y - np.floor(e[0]), self.max_y - np.ceil(e[0])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_boxes_x = np.mean(edges_endpoints_x, axis=1)
        edges_boxes_y = np.mean(edges_endpoints_y, axis=1)

        line_edge_qbinds = np.array([qbind for qbind in line_of_qubits if is_edge_coords(self.coords[qbind])])
        line_vertex_qbinds = np.array(sorted(list(set(line_of_qubits) - set(line_edge_qbinds))))
        non_line_edge_qbinds = np.array(sorted(list(set(self.edge_qbinds) - set(line_edge_qbinds))))
        non_line_vertex_qbinds = np.array(sorted(list(set(self.node_qbinds) - set(line_vertex_qbinds))))

        line_edge_listinds = [np.where(self.edge_qbinds == qbind)[0][0] for qbind in line_edge_qbinds] if len(line_edge_qbinds) > 0 else []
        line_vertex_listinds = [np.where(self.node_qbinds == qbind)[0][0] for qbind in line_vertex_qbinds] if len(line_vertex_qbinds) > 0 else []
        non_line_edge_listinds = [np.where(self.edge_qbinds == qbind)[0][0] for qbind in non_line_edge_qbinds] if len(non_line_edge_qbinds) > 0 else []
        non_line_vertex_listinds = [np.where(self.node_qbinds == qbind)[0][0] for qbind in non_line_vertex_qbinds] if len(non_line_vertex_qbinds) > 0 else []

        fig, ax = plt.subplots(figsize=[scale*self.max_x, scale*self.max_y])
        if color is None: color = "#2f64f7"
        ax.set_aspect('equal')
        for exs, eys in zip(edges_endpoints_x, edges_endpoints_y):
            plt.plot(exs, eys, color="black")
        plt.scatter(vertex_x[line_vertex_listinds], vertex_y[line_vertex_listinds], 350*scale, marker="o", c=color, edgecolors="black", zorder=2)
        plt.scatter(vertex_x[non_line_vertex_listinds], vertex_y[non_line_vertex_listinds], 350*scale, marker="o", c="white", edgecolors="black", zorder=2)
        plt.scatter(edges_boxes_x[line_edge_listinds], edges_boxes_y[line_edge_listinds], 400*scale, marker=(4, 0, 45), c=color, edgecolors="black", zorder=2)
        plt.scatter(edges_boxes_x[non_line_edge_listinds], edges_boxes_y[non_line_edge_listinds], 400*scale, marker=(4, 0, 45), c="white", edgecolors="black", zorder=2)
        if join_line:
            for i in range(len(line_of_qubits)-1):
                first_point_coords = self.coords[line_of_qubits[i]]
                second_point_coords = self.coords[line_of_qubits[i+1]]
                plt.plot([first_point_coords[1], second_point_coords[1]], [self.max_y - first_point_coords[0], self.max_y - second_point_coords[0]], color=color, zorder=1, linewidth=3)
        ttransform = mpl.transforms.Affine2D().translate(0, -1*scale)
        if number_qubits:
            labels = [str(i) for i in range(len(self.coords))]
            for i, (y, x) in enumerate(self.coords):
                text = plt.text(x, self.max_y - y, labels[i], horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale})
                text.set_transform(text.get_transform() + ttransform)
        else:
            line_of_qubit_coords = [self.coords[qbind] for qbind in line_of_qubits]
            for i, (y, x) in enumerate(line_of_qubit_coords):
                textcolor = best_text_color(*mpl.colors.to_rgb(color))
                text = plt.text(x, self.max_y - y, self.coords_to_logical_qb((y, x)), horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale}, color=textcolor)
                text.set_transform(text.get_transform() + ttransform)
        plt.axis("off")
        plt.ylim([-0.2, self.max_y+0.2])
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=300, facecolor="none")
        plt.rcdefaults()

    def plot_lines_of_qubits(self, lines_of_qubits, scale=1.5, number_qubits=False, join_lines=True, colors=None, filepath=""):
        plt.rc("font", family="serif")
        vertex_x = np.array([v[1] for v in sorted(list(self.vertices.keys()))], dtype=int)
        vertex_y = np.array([self.max_y - v[0] for v in sorted(list(self.vertices.keys()))], dtype=int)
        edges_endpoints_x = np.array([[np.floor(e[1]), np.ceil(e[1])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_endpoints_y = np.array([[self.max_y - np.floor(e[0]), self.max_y - np.ceil(e[0])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_boxes_x = np.mean(edges_endpoints_x, axis=1)
        edges_boxes_y = np.mean(edges_endpoints_y, axis=1)

        if isinstance(lines_of_qubits, list) and not all(isinstance(line, list) for line in lines_of_qubits):
            self.plot_line_of_qubits(lines_of_qubits[0], scale, number_qubits, join_lines[0] if isinstance(join_lines, (list, tuple)) else join_lines, plt.get_cmap("Set1")[0] if colors is None else colors[0], filepath)
        else:
            if colors is None:
                cmap = plt.get_cmap("Set1")
                colors = [cmap(i % 8) for i in range(len(lines_of_qubits))]
            join_lines = np.array(join_lines)
            if len(join_lines.shape) == 0:
                join_lines = join_lines*np.ones(len(lines_of_qubits), dtype=bool)
            all_highlighted_qubits = []
            for line_of_qubits in lines_of_qubits:
                all_highlighted_qubits += line_of_qubits
            non_highlighted_qubit_inds = list(set(range(len(self))) - set(all_highlighted_qubits))
            non_highlighted_vertices_ind = np.array([qbind for qbind in non_highlighted_qubit_inds if not is_edge_coords(self.coords[qbind])])
            non_highlighted_edges_ind = np.array([qbind for qbind in non_highlighted_qubit_inds if is_edge_coords(self.coords[qbind])])
            non_highlighted_vertices_listind = [np.where(self.node_qbinds == qbind)[0][0] for qbind in non_highlighted_vertices_ind]
            non_highlighted_edges_listind = [np.where(self.edge_qbinds == qbind)[0][0] for qbind in non_highlighted_edges_ind]
            fig, ax = plt.subplots(figsize=[scale*self.max_x, scale*self.max_y])
            ax.set_aspect('equal')
            for exs, eys in zip(edges_endpoints_x, edges_endpoints_y):
                plt.plot(exs, eys, color="black")
            plt.scatter(vertex_x[non_highlighted_vertices_listind], vertex_y[non_highlighted_vertices_listind], 350*scale, marker="o", color="white", edgecolors="black", zorder=2)
            plt.scatter(edges_boxes_x[non_highlighted_edges_listind], edges_boxes_y[non_highlighted_edges_listind], 400*scale, marker=(4, 0, 45), color="white", edgecolors="black", zorder=2)
            for i, line_of_qubits in enumerate(lines_of_qubits):
                this_line_vertex_inds = np.array([qbind for qbind in line_of_qubits if not is_edge_coords(self.coords[qbind])])
                this_line_edge_inds = np.array([qbind for qbind in line_of_qubits if is_edge_coords(self.coords[qbind])])
                this_line_vertices_listinds = [np.where(self.node_qbinds == qbind)[0][0] for qbind in this_line_vertex_inds]
                this_line_edges_listinds = [np.where(self.edge_qbinds == qbind)[0][0] for qbind in this_line_edge_inds]
                plt.scatter(vertex_x[this_line_vertices_listinds], vertex_y[this_line_vertices_listinds], 350*scale, marker="o", color=colors[i], edgecolors="black", zorder=2)
                plt.scatter(edges_boxes_x[this_line_edges_listinds], edges_boxes_y[this_line_edges_listinds], 400*scale, marker=(4, 0, 45), color=colors[i], edgecolors="black", zorder=2)
                if join_lines[i]:
                    for j in range(len(line_of_qubits)-1):
                        first_point_coords = self.coords[line_of_qubits[j]]
                        second_point_coords = self.coords[line_of_qubits[j+1]]
                        plt.plot([first_point_coords[1], second_point_coords[1]], [self.max_y - first_point_coords[0], self.max_y - second_point_coords[0]], color=colors[i], zorder=1, linewidth=3)
            ttransform = mpl.transforms.Affine2D().translate(0, -1*scale)
            if number_qubits:
                labels = [str(i) for i in range(len(self.coords))]
                for i, (y, x) in enumerate(self.coords):
                    text = plt.text(x, self.max_y - y, labels[i], horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale})
                    text.set_transform(text.get_transform() + ttransform)
            else:
                for i, line_of_qubits in enumerate(lines_of_qubits):
                    this_line_coords = [self.coords[qbind] for qbind in line_of_qubits]
                    for j, (y, x) in enumerate(this_line_coords):
                        textcolor = best_text_color(*mpl.colors.to_rgb(colors[i]))
                        text = plt.text(x, self.max_y - y, self.coords_to_logical_qb((y, x)), horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale}, color=textcolor)
                        text.set_transform(text.get_transform() + ttransform)
            plt.axis("off")
            plt.ylim([-0.2, self.max_y+0.2])
            plt.tight_layout()
            if filepath:
                plt.savefig(filepath, dpi=300, facecolor="none")
            plt.rcdefaults()

    def plot_highlighted_qubits(self, qubits, scale=1.5, number_qubits=False, colors=None, filepath=""):
        plt.rc("font", family="serif")
        vertex_x = np.array([v[1] for v in sorted(list(self.vertices.keys()))], dtype=int)
        vertex_y = np.array([self.max_y - v[0] for v in sorted(list(self.vertices.keys()))], dtype=int)
        edges_endpoints_x = np.array([[np.floor(e[1]), np.ceil(e[1])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_endpoints_y = np.array([[self.max_y - np.floor(e[0]), self.max_y - np.ceil(e[0])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_boxes_x = np.mean(edges_endpoints_x, axis=1)
        edges_boxes_y = np.mean(edges_endpoints_y, axis=1)

        non_highlighted_qubit_inds = list(set(range(len(self))) - set(np.array(qubits).tolist()))
        highlighted_nodes_inds = [np.where(np.equal(self.node_qbinds, qbind))[0][0] for qbind in qubits if not is_edge_coords(self.coords[qbind]) and len(np.where(np.equal(self.node_qbinds, qbind))) > 0]
        highlighted_edges_inds = [np.where(np.equal(self.edge_qbinds, qbind))[0][0] for qbind in qubits if is_edge_coords(self.coords[qbind]) and len(np.where(np.equal(self.edge_qbinds, qbind))) > 0]
        non_highlighted_nodes_inds = [np.where(np.equal(self.node_qbinds, qbind))[0][0] for qbind in non_highlighted_qubit_inds if not is_edge_coords(self.coords[qbind]) and len(np.where(np.equal(self.node_qbinds, qbind))) > 0]
        non_highlighted_edges_inds = [np.where(np.equal(self.edge_qbinds, qbind))[0][0] for qbind in non_highlighted_qubit_inds if is_edge_coords(self.coords[qbind]) and len(np.where(np.equal(self.edge_qbinds, qbind))) > 0]

        highlighted_nodes_color_inds = [i for i, qbind in enumerate(qubits) if not is_edge_coords(self.coords[qbind])]
        highlighted_edges_color_inds = [i for i, qbind in enumerate(qubits) if is_edge_coords(self.coords[qbind])]

        fig, ax = plt.subplots(figsize=[scale*self.max_x, scale*self.max_y])
        if colors is None:
            cmap = plt.get_cmap("Set2")
            colors = np.array([cmap((i/8)%8) for i in range(len(qubits))])
        else:
            colors = np.array(colors, dtype=object)
        ax.set_aspect('equal')
        for exs, eys in zip(edges_endpoints_x, edges_endpoints_y):
            plt.plot(exs, eys, color="black")
        plt.scatter(vertex_x[highlighted_nodes_inds], vertex_y[highlighted_nodes_inds], 350*scale, marker="o", c=colors[highlighted_nodes_color_inds], edgecolors="black", zorder=2)
        plt.scatter(vertex_x[non_highlighted_nodes_inds], vertex_y[non_highlighted_nodes_inds], 350*scale, marker="o", c="white", edgecolors="black", zorder=2)
        plt.scatter(edges_boxes_x[highlighted_edges_inds], edges_boxes_y[highlighted_edges_inds], 400*scale, marker=(4, 0, 45), c=colors[highlighted_edges_color_inds], edgecolors="black", zorder=2)
        plt.scatter(edges_boxes_x[non_highlighted_edges_inds], edges_boxes_y[non_highlighted_edges_inds], 400*scale, marker=(4, 0, 45), c="white", edgecolors="black", zorder=2)
        ttransform = mpl.transforms.Affine2D().translate(0, -1*scale)
        if number_qubits:
            labels = [str(i) for i in range(len(self.coords))]
            for i, (y, x) in enumerate(self.coords):
                text = plt.text(x, self.max_y - y, labels[i], horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale})
                text.set_transform(text.get_transform() + ttransform)
        else:
            for i, qbind in enumerate(qubits):
                y, x = self.coords[qbind]
                textcolor = best_text_color(*mpl.colors.to_rgb(colors[i]))
                text = plt.text(x, self.max_y - y, self.coords_to_logical_qb((y, x)), horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale}, color=textcolor)
                text.set_transform(text.get_transform() + ttransform)
        plt.axis("off")
        plt.ylim([-0.2, self.max_y+0.2])
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=300, facecolor="none")
        plt.rcdefaults()

    def complete_string_with_matter(self, string_coords_qubits):
        if isinstance(string_coords_qubits[0], int):
            string_coords = [self.coords[qbind] for qbind in string_coords_qubits]
        else:
            string_coords = string_coords_qubits 
        string_coords = np.array(string_coords)
        if not ((np.equal(np.round(string_coords[0]), string_coords[0]).all()) or (np.equal(np.round(string_coords[-1]), string_coords[-1]).all())):
            raise ValueError("The string must start and end in a matter qubit")
        complete_string_coords = [tuple(string_coords[0])]
        for coords in string_coords:
            coords = array_of_coords_to_tuple(coords)
            if is_edge_coords(coords):
                connected_nodes = self.nodes_connected_to_edge(coords)
                for node in connected_nodes:
                    if node not in complete_string_coords:
                        complete_string_coords.append(coords)
                        complete_string_coords.append(node)
        return np.array(complete_string_coords)             

def get_pair_qubit_inds(strings):
    pair_qubit_inds = []
    for corr_type, string in enumerate(strings):
        if corr_type == 0:
            qubit_ind1 = string[0]
            for i in range((len(string)-1)//2):
                qubit_ind2 = string[i+1]
                pair_qubit_inds.append((qubit_ind1, qubit_ind2))
        elif corr_type == 1:
            qubit_ind1 = string[0]
            for i in range(len(string)-1):
                qubit_ind2 = string[i+1]
                pair_qubit_inds.append((qubit_ind1, qubit_ind2))
        else:
            qubit_ind1 = string[len(string)//2]
            for i in range((len(string)-1)//2):
                qubit_ind2 = string[len(string)//2+i+1]
                pair_qubit_inds.append((qubit_ind1, qubit_ind2))
    return pair_qubit_inds

def is_edge_coords(coords):
    if type(coords) == tuple:
        return (coords[0] % 1 != 0) or (coords[1] % 1 != 0)
    else:
        try:
            coords = np.array(coords)
            return ~np.equal(coords.sum(axis=1) % 1, 0)
        except np.AxisError:
            raise np.AxisError("Coords is not a valid tuple or N x 2 array")

def plot_layout_over_device(backend, layout, scale=1):
    backend_coords = HeavyHexLattice.get_backend_coordinates(backend)
    backend_edges = []
    edges_colors = []
    backend_vertices = []
    vertices_colors = []
    for i, coord in enumerate(backend_coords):
        if coord[1] % 2 == 1 or coord[0] % 2 == 1:
            backend_edges.append(coord)
            edges_colors.append("#2f64f7" if i in layout else "white")
        else:
            vertices_colors.append("#2f64f7" if i in layout else "white")
            backend_vertices.append(coord)
    max_x = max([coords[1] for coords in backend_coords])
    max_y = max([coords[0] for coords in backend_coords])
    vertices_x = np.array([v[1] for v in backend_vertices], dtype=int)
    vertices_y = np.array([max_y - v[0] for v in backend_vertices], dtype=int)
    edges_endpoints_x = []
    edges_endpoints_y = []
    edges_boxes_x = []
    edges_boxes_y = []
    for edge in backend_edges:
        if edge[0] % 2 == 0:
            edges_endpoints_x.append([edge[1]-1, edge[1]+(not edge[1] == max_x)])
            edges_endpoints_y.append([max_y - edge[0], max_y - edge[0]])
        else:
            edges_endpoints_x.append([edge[1], edge[1]])
            edges_endpoints_y.append([max_y - (edge[0] - 1), max_y - (edge[0] + 1)])
        edges_boxes_x.append(edge[1])
        edges_boxes_y.append(max_y - edge[0])

    fig, ax = plt.subplots(figsize=[scale*max_x, scale*max_y])
    ax.set_aspect('equal')
    for exs, eys in zip(edges_endpoints_x, edges_endpoints_y):
        plt.plot(exs, eys, color="black")
    plt.scatter(vertices_x, vertices_y, 600*scale, vertices_colors, marker="o", edgecolors="black", zorder=2)
    plt.scatter(edges_boxes_x, edges_boxes_y, 700*scale, edges_colors, marker=(4, 0, 45), edgecolors="black", zorder=2)
    ttransform = mpl.transforms.Affine2D().translate(0, -1*scale)
    for qbind, (y, x) in enumerate(backend_coords):
        if qbind in layout:
            layout_ind = layout.index(qbind)
            text = plt.text(x, max_y - y, str(layout_ind), color="white", fontsize=15*scale, ha='center', va='center')
            text.set_transform(text.get_transform() + ttransform)
    plt.axis("off")
    plt.ylim([-0.2, max_y+0.2])
    plt.tight_layout()
    plt.rcdefaults()

def array_of_coords_to_tuple(coords):
    try:
        if len(coords.shape) == 2:
            if coords.shape[1] != 2:
                raise ValueError("An array of coordinates must be a n x 2 matrix")
            to_return = []
            for coord in coords:
                to_return.append(tuple((coord[0] if coord[0] % 1 != 0 else int(coord[0]), coord[1] if coord[1] % 1 != 0 else int(coord[1]))))
            return to_return
        elif len(coords.shape) == 1:
            if coords.shape[0] != 2:
                raise ValueError("An array of coordinates must be a n x 2 matrix")
            return tuple((coords[0] if coords[0] % 1 != 0 else int(coords[0]), coords[1] if coords[1] % 1 != 0 else int(coords[1])))
    except AttributeError:
        raise ValueError("coords is not an array")
    
def sort_layouts_by_func(layouts, sort_func):
    """Sorts the layouts by a function
    Arguments:
        layouts:
            List of layouts to be sorted
        sort_func:
            Function to sort the layouts by, 
            it must take a 2D array of layouts and return a 1D array of costs
    """
    if not callable(sort_func):
        raise ValueError("sort_func must be a callable function")
    try:
        layouts = np.array(layouts, dtype=int)
        if layouts.ndim != 2:
            raise ValueError("Layouts must be a 2D array")
    except ValueError:
        raise ValueError("Layouts must be a 2D array")
    if np.sum(np.unique(layouts, axis=1) != layouts):
        raise ValueError("Layouts must be a 2D array with unique integer columns")
    try:
        costs = sort_func(layouts)
    except ValueError:
        raise ValueError("sort_func must take a 2D array as input and return a 1D array of costs")
    reorder_mask = np.argsort(costs)
    return layouts[reorder_mask, :]

def measurement_decoherence_layout_cost(layouts, backend):
    """Returns the cost of layouts based on T1/T1 decoherence times and measurement errors
    Arguments:
        layouts:
            List of layouts to be sorted
        backend:
            Device backend
    """
    t1s = np.array([qubitprops[0].value for qubitprops in backend.properties().qubits])
    t2s = np.array([qubitprops[1].value for qubitprops in backend.properties().qubits])
    readout_errors = np.array([qubitprops[2].value for qubitprops in backend.properties().qubits])
    each_qb_t1 = t1s[layouts]
    each_qb_t2 = t2s[layouts]
    each_qb_readout = readout_errors[layouts]
    connection_errors = []
    for i, layout in enumerate(layouts):
        this_cz_gate_errors = []
        for gate in backend.properties().gates:
            if gate.gate == "cz" and (gate.qubits[0] in layout) and (gate.qubits[1] in layout):
                this_error = gate.parameters[0].value
                this_cz_gate_errors.append(gate.parameters[0].value)
        connection_errors.append(this_cz_gate_errors)
    connection_errors = np.array(connection_errors)
    t1_costs = np.mean((np.max(t1s) - each_qb_t1)/(np.max(t1s) - np.min(t1s)), axis=1)
    t2_costs = np.mean((np.max(t2s) - each_qb_t2)/(np.max(t2s) - np.min(t2s)), axis=1)
    readout_costs = np.mean((each_qb_readout - np.min(readout_errors))/(np.max(readout_errors) - np.min(readout_errors)), axis=1)
    zero_connection_costs_mask = ~np.isclose(connection_errors, 0)
    connection_costs = np.mean((connection_errors[zero_connection_costs_mask] - np.min(connection_errors[zero_connection_costs_mask]))/(np.max(connection_errors[zero_connection_costs_mask]) - np.min(connection_errors[zero_connection_costs_mask])))
    return t1_costs + t2_costs + readout_costs + connection_costs