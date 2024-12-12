import numpy as np
import math


class Node:
    def __init__(self, x,y,is_bc=False):
        self.x = x
        self.y = y
        self.is_bc = is_bc          # flaga okreslajaca czy wezel jest brzegowy


class Element:
    def __init__(self, nodes, element_type, integration_order):
        self.nodes = nodes
        self.H_local = None
        self.H_bc = None
        self.P_local = None
        self.element_type = element_type
        self.edges = self.create_edges(integration_order)

    def create_edges(self, integration_order):
        if self.element_type == "4-node":
            return [
                Edge(self.nodes[0], self.nodes[1], integration_order),
                Edge(self.nodes[1], self.nodes[2], integration_order),
                Edge(self.nodes[2], self.nodes[3], integration_order),
                Edge(self.nodes[3], self.nodes[0], integration_order),
            ]
        elif self.element_type == "9-node":
            return [
                Edge(self.nodes[0], self.nodes[1], integration_order),
                Edge(self.nodes[1], self.nodes[5], integration_order),
                Edge(self.nodes[5], self.nodes[2], integration_order),
                Edge(self.nodes[2], self.nodes[6], integration_order),
                Edge(self.nodes[6], self.nodes[3], integration_order),
                Edge(self.nodes[3], self.nodes[7], integration_order),
                Edge(self.nodes[7], self.nodes[0], integration_order),
                Edge(self.nodes[4], self.nodes[8], integration_order),
            ]

    def shape_functions(self, xi, eta):
        if self.element_type == "4-node":
            return np.array([
                (1 - xi) * (1 - eta) / 4,
                (1 + xi) * (1 - eta) / 4,
                (1 + xi) * (1 + eta) / 4,
                (1 - xi) * (1 + eta) / 4,
                ])
        elif self.element_type == "9-node":
            return np.array([
                1 / 4 * (1 - xi) * (1 - eta) * xi * eta,
                -1 / 4 * (1 + xi) * (1 - eta) * xi * eta,
                1 / 4 * (1 + xi) * (1 + eta) * xi * eta,
                -1 / 4 * (1 - xi) * (1 + eta) * xi * eta,
                -1 / 2 * (1 - xi**2) * (1 - eta) * eta,
                1 / 2 * (1 + xi) * (1 - eta**2) * xi,
                -1 / 2 * (1 - xi**2) * (1 + eta) * eta,
                -1 / 2 * (1 - xi) * (1 - eta**2) * xi,
                (1 - xi**2) * (1 - eta**2),
                ])

    def shape_function_derivatives(self, xi, eta):
        if self.element_type == "4-node":
            dN_dxi = np.array([
                -(1 - eta) / 4,
                (1 - eta) / 4,
                (1 + eta) / 4,
                -(1 + eta) / 4,
                ])
            dN_deta = np.array([
                -(1 - xi) / 4,
                -(1 + xi) / 4,
                (1 + xi) / 4,
                (1 - xi) / 4,
                ])
        elif self.element_type == "9-node":
            dN_dxi = np.array([
                1 / 4 * (1 - eta) * (2 * xi * eta - eta),
                -1 / 4 * (1 - eta) * (2 * xi * eta + eta),
                1 / 4 * (1 + eta) * (2 * xi * eta + eta),
                -1 / 4 * (1 + eta) * (2 * xi * eta - eta),
                -1 / 2 * 2 * xi * (1 - eta) * eta,
                1 / 2 * (1 - eta**2) * (2 * xi + 1),
                -1 / 2 * 2 * xi * (1 + eta) * eta,
                -1 / 2 * (1 - eta**2) * (2 * xi - 1),
                -2 * xi * (1 - eta**2),
                ])
            dN_deta = np.array([
                1 / 4 * (1 - xi) * (xi * 2 * eta - xi),
                -1 / 4 * (1 + xi) * (xi * 2 * eta + xi),
                1 / 4 * (1 + xi) * (xi * 2 * eta + xi),
                -1 / 4 * (1 - xi) * (xi * 2 * eta - xi),
                -1 / 2 * (1 - xi**2) * (2 * eta - 1),
                1 / 2 * (1 + xi) * (-2 * eta),
                -1 / 2 * (1 - xi**2) * (2 * eta + 1),
                -1 / 2 * (1 - xi) * (-2 * eta),
                -2 * eta * (1 - xi**2),
                ])
        return dN_dxi, dN_deta

class Edge:
    def __init__(self, node1, node2, integration_order):
        self.node1 = node1
        self.node2 = node2
        self.integration_order = integration_order
        self.integration_points = None
        self.weights = None
        self.calculate_integration_properties()

    def calculate_integration_properties(self):
        self.integration_points, self.weights = integration_scheme_1D(self.integration_order)

    def length(self):
        return np.sqrt((self.node2.x - self.node1.x)**2 + (self.node2.y - self.node1.y)**2)



def compute_jacobian(element, xi, eta):
    dN_dxi, dN_deta = element.shape_function_derivatives(xi, eta)
    J = np.zeros((2, 2))
    for i, node in enumerate(element.nodes):
        J[0, 0] += dN_dxi[i] * node.x  # dN/dxi * x_i
        J[0, 1] += dN_dxi[i] * node.y  # dN/dxi * y_i
        J[1, 0] += dN_deta[i] * node.x  # dN/deta * x_i
        J[1, 1] += dN_deta[i] * node.y  # dN/deta * y_i
    return J


def determinant_of_jacobian(J):
    return np.linalg.det(J)


def inverse_of_jacobian(J):
    return np.linalg.inv(J)


# Funkcja do obliczania macierzy lokalnej H
def compute_local_H(element, k, integration_order):
    num_nodes = len(element.nodes)
    H = np.zeros((num_nodes, num_nodes))

    # print(f"Rozmiar macierzy H: {H.shape}")

    # Pobranie punktów i wag Gaussa
    integration_points, weights = integration_scheme(integration_order)

    for i, (xi, eta) in enumerate(integration_points):
        weightX = weights[i % len(weights)]  # Dopasowanie wag do punktów
        weightY = weights[i // len(weights)] # Dopasowanie wag do punktów
        # Obliczenie Jakobianu
        J = compute_jacobian(element, xi, eta)
        detJ = determinant_of_jacobian(J)
        invJ = inverse_of_jacobian(J)

        # # Wyświetlenie Jakobianu
        # print(f"Punkt Gaussa {i+1}: xi={xi}, eta={eta}")
        # print(f"Jakobian:\n{J}\nWyznacznik: {detJ}\nMacierz odwrotna:\n{invJ}")

        # Pochodne funkcji kształtu względem (xi, eta)
        dN_dxi, dN_deta = element.shape_function_derivatives(xi, eta)

        # Pochodne funkcji kształtu względem (x, y)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        # # Wyświetlenie pochodnych funkcji kształtu
        # print(f"dN{i+1}_dx: {dN_dx}")
        # print(f"dN{i+1}_dy: {dN_dy}")
        #
        # # Wyświetlenie pochodnych funkcji kształtu dla każdego węzła
        # for n in range(len(dN_dx)):
        #     print(f"dN{n+1}_dx: {dN_dx[n]}")
        #     print(f"dN{n+1}_dy: {dN_dy[n]}")

        # Lokalna macierz H w punkcie Gaussa
        H_local = k * (np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy)) * detJ * weightX * weightY

        # # Rozmiar i zawartość H_local
        # print(f"Rozmiar H_local: {H_local.shape}")
        # print(f"H_local dla punktu {i+1}:\n{H_local}")

        # Dodanie macierzy lokalnej do macierzy globalnej
        H += H_local

    # # Końcowa macierz H
    # print(f"Końcowa macierz H:\n{H}")

    element.H_local = H

    return H


def compute_Hbc(element, alpha):
    num_nodes = len(element.nodes)
    Hbc = np.zeros((num_nodes, num_nodes))

    for edge in element.edges:
        if not (edge.node1.is_bc and edge.node2.is_bc):
            continue

        edge_length = edge.length()
        detJ = edge_length / 2

        for i, xi in enumerate(edge.integration_points):
            weight = edge.weights[i]
            N = [(1 - xi) / 2, (1 + xi) / 2]  # Funkcje kształtu dla krawędzi

            # Obliczanie macierzy Hbc dla krawędzi
            Hbc_edge = alpha * np.outer(N, N) * weight * detJ

            # Indeksy węzłów w macierzy
            local_indices = [element.nodes.index(edge.node1), element.nodes.index(edge.node2)]

            # Dodanie wyników
            for a, global_a in enumerate(local_indices):
                for b, global_b in enumerate(local_indices):
                    Hbc[global_a, global_b] += Hbc_edge[a, b]

    element.H_bc = Hbc
    return Hbc

# Funkcja do obliczania wektora P
def compute_local_P(element, alpha, t_env):
    num_nodes = len(element.nodes)
    P = np.zeros(num_nodes)

    for edge in element.edges:
        if not (edge.node1.is_bc and edge.node2.is_bc):
            continue

        edge_length = edge.length()
        detJ = edge_length / 2

        for i, xi in enumerate(edge.integration_points):
            weight = edge.weights[i]
            N = [(1 - xi) / 2, (1 + xi) / 2]  # Shape functions for the edge

            # Compute the contribution to P for the edge
            P_edge = alpha * t_env * np.array(N) * weight * detJ

            # Indices in the global system
            local_indices = [element.nodes.index(edge.node1), element.nodes.index(edge.node2)]

            # Add contributions to the local P vector
            for a, global_a in enumerate(local_indices):
                P[global_a] += P_edge[a]

    element.P = P
    return P

# Funkcja do agregacji
def aggregate_to_global_H(global_H, local_H, global_indices):
    for local_i, global_i in enumerate(global_indices):
        for local_j, global_j in enumerate(global_indices):
            global_H[global_i,global_j] += local_H[local_i,local_j]

# Funkcja do agregacji wektora P
def aggregate_to_global_P(global_P, local_P, global_indices):
    for local_i, global_i in enumerate(global_indices):
        global_P[global_i] += local_P[local_i]


def integration_scheme_2():
    points = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    weights = [1, 1]
    xi_eta = [(xi, eta) for xi in points for eta in points]
    return xi_eta, weights


def integration_scheme_3():
    points = [-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]
    weights = [5 / 9, 8 / 9, 5 / 9]
    xi_eta = [(xi, eta) for xi in points for eta in points]
    return xi_eta, weights


def integration_scheme_4():
    points = [-0.861136, -0.339981, 0.339981, 0.861136]
    weights = [0.347855, 0.652145, 0.652145, 0.347855]
    xi_eta = [(xi, eta) for xi in points for eta in points]
    return xi_eta, weights


def integration_scheme(order):
    if order == 2:
        return integration_scheme_2()
    elif order == 3:
        return integration_scheme_3()
    elif order == 4:
        return integration_scheme_4()
    else:
        raise ValueError(f"Nieobsługiwany schemat całkowania: {order}")

# potrzebne do krawedzi
def integration_scheme_1D(order):
    if order == 2:
        points = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
        weights = [1, 1]
    elif order == 3:
        points = [-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]
        weights = [5 / 9, 8 / 9, 5 / 9]
    elif order == 4:
        points = [-0.861136, -0.339981, 0.339981, 0.861136]
        weights = [0.347855, 0.652145, 0.652145, 0.347855]
    else:
        raise ValueError(f"Nieobsługiwany schemat całkowania 1D: {order}")
    return points, weights


# Odczytywanie wspolrzednych wezlow, struktura pliku:
# x1 y1
# x2 y2
def read_coordinates_from_file(file_path):
    with open(file_path, 'r') as file:
        coords = []
        for line in file:
            x, y = map(float, line.split())
            coords.append((x, y))
    return coords

# Funkcje do odczytania wartosci z pliku o strukturze takiej jak w Test1_4_4.txt
# Odczyt zmiennych z naglowka
def read_header(file):
    data = {}
    for line in file:
        if line.strip().startswith("*"):
            break
        parts = line.split()
        key = parts[0]
        value = parts[-1]
        data[key] = float(value) if "." in value else int(value)
    return data

def read_nodes(file):
    nodes = []
    for line in file:
        if line.strip().startswith("*"):
            break
        _, x, y = map(float, line.split(","))
        nodes.append((x, y))
    return nodes

def read_elements(file):
    elements = []
    for line in file:
        if line.strip().startswith("*"):
            break
        _, *node_ids = map(int, line.split(","))
        elements.append(node_ids)
    return elements

def read_bc(file):
    bc_nodes = []
    for line in file:
        if line.strip().startswith("*"):
            break
        bc_nodes.extend(map(int, line.split(",")))
    return bc_nodes

# Parsowanie calego pliku
def parse_mesh_file(file_path):
    with open(file_path, "r") as file:
        header = read_header(file)
        node_data = read_nodes(file)
        element_data = read_elements(file)
        bc_nodes = set(read_bc(file))

    return {
        "header": header,
        "node_data": node_data,
        "element_data": element_data,
        "bc_nodes": bc_nodes,
    }

def create_nodes_and_elements(parsed_data, element_type, integration_order):
    nodes = [
        Node(x, y, i + 1 in parsed_data["bc_nodes"])
        for i, (x, y) in enumerate(parsed_data["node_data"])
    ]

    elements = [
        Element([nodes[i - 1] for i in element], element_type, integration_order)
        for element in parsed_data["element_data"]
    ]

    return nodes, elements

# if __name__ == "__main__":
#     # data = parse_mesh_file("Test1_4_4.txt")
#     data = parse_mesh_file("Test2_4_4_MixGrid.txt")
#     integration_order = 2
#     nodes, elements = create_nodes_and_elements(data, element_type="4-node", integration_order=integration_order)
#
#     alpha = data["header"]["Alfa"]
#     k = data["header"]["Conductivity"]
#
#     print("=== Testowanie macierzy Hbc ===")
#     for element in elements:
#         Hbc = compute_Hbc(element, alpha)
#         print(f"Macierz Hbc dla elementu:\n{Hbc}")
#
#     print("\n=== Testowanie macierzy H lokalnych ===")
#     for element in elements:
#         H_local = compute_local_H(element, k, integration_order)
#         print(f"Macierz H lokalna dla elementu:\n{H_local}")
#
#     print("\n=== Testowanie macierzy globalnej ===")
#     num_global_nodes = data["header"]["Nodes"]
#     global_H = np.zeros((num_global_nodes, num_global_nodes))
#
#     for element, element_node_ids in zip(elements, data["element_data"]):
#         H_local = compute_local_H(element, k, integration_order)
#         Hbc = compute_Hbc(element, alpha)
#         H_total = H_local + Hbc
#
#         global_indices = [node_id - 1 for node_id in element_node_ids]
#         aggregate_to_global_H(global_H, H_total, global_indices)
#
#     print("Macierz globalna H (z uwzględnieniem Hbc):")
#     print(global_H)
if __name__ == "__main__":
    data = parse_mesh_file("Test2_4_4_MixGrid.txt")
    # data = parse_mesh_file("Test1_4_4.txt")
    # data = parse_mesh_file("H_test.txt")
    integration_order = 2
    nodes, elements = create_nodes_and_elements(data, element_type="4-node", integration_order=integration_order)

    alpha = data["header"]["Alfa"]
    k = data["header"]["Conductivity"]
    Tot = data["header"]["Tot"]

    global_P = np.zeros(len(nodes))

    # print("\n=== Macierze H lokalne ===")
    # for idx, element in enumerate(elements, start=1):
    #     H_local = compute_local_H(element, k, integration_order)
    #     print(f"H dla elementu - {idx}")
    #     for row in H_local:
    #         print(" ".join(f"{value:.5f}" for value in row))

    print("\n=== Macierze H całkowite (H_total) ===")
    for idx, (element, element_node_ids) in enumerate(zip(elements, data["element_data"]), start=1):
        H_local = compute_local_H(element, k, integration_order)
        Hbc = compute_Hbc(element, alpha)
        H_total = H_local + Hbc


        P_local = compute_local_P(element, alpha, Tot)
        element.P_local = P_local

        print(f"H całkowita (H_total) dla elementu - {idx}")
        for row in H_total:
            print(" ".join(f"{value:.5f}" for value in row))

        print("Wektor P lokalny:")
        print(" ".join(f"{value:.1f}" for value in P_local))

    print("\n=== Testowanie macierzy globalnej ===")
    num_global_nodes = data["header"]["Nodes"]
    global_H = np.zeros((num_global_nodes, num_global_nodes))

    for element, element_node_ids in zip(elements, data["element_data"]):
        H_local = compute_local_H(element, k, integration_order)
        Hbc = compute_Hbc(element, alpha)
        H_total = H_local + Hbc
        P_local = compute_local_P(element, alpha, Tot)
        element.P_local = P_local

        global_indices = [node_id - 1 for node_id in element_node_ids]
        aggregate_to_global_P(global_P, P_local, global_indices)
        aggregate_to_global_H(global_H, H_total, global_indices)


    # Wyświetlenie ostatecznej macierzy globalnej H
    print("Macierz globalna H:")
    for row in global_H:
        print(" ".join(f"{value:.5f}" for value in row))

    # Wyświetlenie wektora globalnego P
    print("Globalny wektor P:")
    print(" ".join(f"{value:.1f}" for value in global_P))
