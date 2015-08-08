## hyperelasticity02.py
## These are my changes in implementing hyperelasticity
## using FEniCS 
## 
## All credit should be given to the FEniCS team, Johan Hake, and C++ demo by Harish Narayanan
## for the original implementation, which I'll copy and borrow from liberally 
## 
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150801
##                                                                          
## This program, along with all its code, is free software; 
## you can redistribute it and/or modify  
## it under the terms of the GNU General Public License as published by                
## the Free Software Foundation; either version 2 of the License, or        
## (at your option) any later version.                               
##                                                                
## This program is distributed in the hope that it will be useful,             
## but WITHOUT ANY WARRANTY; without even the implied warranty of                      
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the    
## GNU General Public License for more details.                      
##                                                                       
## You can have received a copy of the GNU General Public License              
## along with this program; if not, write to the Free Software Foundation, Inc.,  
## S1 Franklin Street, Fifth Floor, Boston, MA                      
## 02110-1301, USA                                              
##                                                
## Governing the ethics of using this program, I default to the Caltech Honor Code:  
## ``No member of the Caltech community shall take unfair advantage of               
## any other member of the Caltech community.''                       
##                                                                                  
## Donate and support my scientific and engineering efforts here:
## ernestyalumni.tilt.com                                                         
##                                                                                    
## Facebook     : ernestyalumni                                                       
## linkedin     : ernestyalumni                                                  
## Tilt/Open    : ernestyalumni                                                   
## twitter      : ernestyalumni                                               
## youtube      : ernestyalumni                                                 
## wordpress    : ernestyalumni                                      
##  
################################################################################
## 
## Original preface:
""" This demo program solves a hyperelastic problem. It is implemented
in Python by Johan Hake following the C++ demo by Harish Narayanan"""

# Copyright (C) 2008-2010 Johan Hake and Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Harish Narayanan 2009
# Modified by Anders Logg 2011
#
# First added:  2009-10-11
# Last changed: 2012-11-12

# Begin demo

from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = BoxMesh(0,0,0,0.1,0.1,2.0,10,10,200)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomains
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.1)
bottom =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.1)
zleft  = CompiledSubDomain("near(x[2], side) && on_boundary", side = 0.0) 
zright = CompiledSubDomain("near(x[2], side) && on_boundary", side = 2.0)

# Define Dirichlet boundary (x = 0 or x = 1)
#c = Expression(("scale*(x0 + (x[0] - x0)*cos(theta) - (x[1] - y0)*sin(theta) - x[0])",
#                "scale*(y0 + (x[0] - x0)*sin(theta) + (x[1] - y0)*cos(theta) - x[1])", 
#                "scale*(z0-x[2])"
#                ),
#               scale = 0.5, x0 = 0.05, y0 = 0.05, theta = pi/3)
#r = Expression(("scale*0.0",
#                "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
#                "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"
#                ),
#                scale = 0.5, x0 = 0.05, y0 = 0.05, theta = pi/3)

clamp = Expression(("0.0","0.0","0.0"))

bcl = DirichletBC(V, clamp, zleft)
bcr = DirichletBC(V, clamp, zright)

#bcl = DirichletBC(V, c, zleft)
#bcr = DirichletBC(V, r, zright)
bcs = [bcl, bcr]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.2224111, 0.0))  # Body force per unit volume   # **6
T  = Constant((0.0,  0.0, 1.5169))  # Traction force on the boundary  # **8

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 200.0*0.1, 0.29 # 200 GPa, Gigapascal, for steel  #**9
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem
solve(F == 0, u, bcs, J=J,
      form_compiler_parameters=ffc_options)

# Save solution in VTK format
file = File("displacement2.pvd");
file << u;

# Plot and hold solution
plot(u, mode = "displacement", interactive = True)
