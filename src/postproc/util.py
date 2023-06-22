import numpy as np


def find_fermi_crossing(k_points_fermi,k_path):
    fermi_crossing_ind = []
    for j in range(len(k_path.nk_seg)):
        dist_to_fermi = []
        for i in range(k_path.nk_seg[j]):
            ind = i + k_path.cind[j]
            k_point_k_path = np.array([k_path.k_val[0][ind], k_path.k_val[1][ind]])

            dist_to_fermi.append(np.min(np.linalg.norm(k_points_fermi - k_point_k_path, axis=1)))
        fermi_crossing_ind.append(np.argmin(dist_to_fermi) + k_path.cind[j])
    return fermi_crossing_ind