# coding: utf-8
""" daeger pulmovista .eit file reading and dynamic EIT imaging """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.io import DAEGER_EIT
import pyeit.eit.greit as greit
import pyeit.eit.protocol as protocol

""" 0. construct mesh (PyEITMesh dataset) """
mesh_obj = mesh.create(16, h0=0.1)
# extract node, element, sigma
pts = mesh_obj.node
tri = mesh_obj.element

# plot the mesh, visualize the electrodes locations
fig, ax = plt.subplots(figsize=(6, 6))
ax.triplot(pts[:, 0], pts[:, 1], tri, linewidth=1)
ax.plot(pts[mesh_obj.el_pos, 0], pts[mesh_obj.el_pos, 1], "ro")
for i, el in enumerate(mesh_obj.el_pos):
    ax.text(pts[el, 0], pts[el, 1], str(i + 1), fontsize=12)
ax.axis("equal")
ax.axis([-1.2, 1.2, -1.2, 1.2])

""" 1. load data """
fname = "./ID_SC_10_60.eit"
model = DAEGER_EIT(fname)
data = model.load()
v0 = data[10]
v1 = data[24]

# plot transfer impedance
fig, ax = plt.subplots(figsize=(6, 4))
ati = np.mean(data, axis=1)
ax.plot(ati)
ax.grid(True)
ax.set_xlabel("Frames 1/20 (s)")
ax.set_ylabel("Averaged Impedances")
fig.subplots_adjust(top=0.95, bottom=0.15, left=0.175, right=0.95)

""" 2. measurement protocol """
protocol_obj = protocol.create(
    mesh_obj.n_el, dist_exc=1, step_meas=1, parser_meas="fmmu"
)

""" 3. JAC solver """
# Note: if the jac and the real-problem are generated using the same mesh,
# then, jac_normalized in JAC and data normalization in solve are not needed.
# However, when you generate jac from a known mesh, but in real-problem
# (mostly) the shape and the electrode positions are not exactly the same
# as in mesh generating the jac, then both JAC and data must be normalized.
eit = greit.GREIT(mesh_obj, protocol_obj)
eit.setup(p=0.50, lamb=0.01, n=32, s=20, ratio=0.05, jac_normalized=True)
ds = eit.solve(v1, v0, normalize=True)
x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

# plot EIT imaging
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis("equal")
# fig.set_size_inches(6, 4)
# fig.savefig('./figs/daeger_pulmovista_eit.png', dpi=200)
plt.show()
