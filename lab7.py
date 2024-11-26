import numpy as np
import math


class Node:
    def __init__(self, x,y,is_bc=False):
        self.x = x
        self.y = y
        self.is_bc = is_bc          # flaga okreslajaca czy wezel jest brzegowy


class Element:
    def __init__(self, nodes, element_type):
        self.nodes = nodes
        self.H_local = None
        self.H_bc = None
        self.element_type = element_type

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

def create_nodes_and_elements(parsed_data, element_type):
    # Tworzenie węzłów
    nodes = [
        Node(x, y, i + 1 in parsed_data["bc_nodes"])
        for i, (x, y) in enumerate(parsed_data["node_data"])
    ]

    # Tworzenie elementów
    elements = [
        Element([nodes[i - 1] for i in element], element_type)
        for element in parsed_data["element_data"]
    ]

    return nodes, elements
