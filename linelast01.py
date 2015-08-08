## linelast01.py
## This is my implementation of linear elasticity using FEniCS
## The reference that I used was 
## A. Logg, K.-A. Mardal, G. N. Wells et al. (2012). Automated Solution of Differential Equations by the Finite Element Method, Springer. [doi:10.1007/978-3-642-23099-8]  
## particularly Chapter 26 Applications to Solid Mechanics 
## by Kristian Oelgaard and Garth Wells
############################################################################ 
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                            
##                                                            
## 20150806
##                                                                               
## This program, along with all its code, is free software; you can redistribute 
## it and/or modify it under the terms of the GNU General Public License as 
## published by the Free Software Foundation; either version 2 of the License, or   
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
## Donate and support my scientific and engineering efforts here                
## ernestyalumni.tilt.com                                                      
##                                                                              
## Facebook     : ernestyalumni                                                   
## linkedin     : ernestyalumni                                                    
## Tilt/Open    : ernestyalumni                                                    
## twitter      : ernestyalumni                                                   
## youtube      : ernestyalumni                                                   
## wordpress    : ernestyalumni                                                    
##  
############################################################################ 

import numpy as np

from dolfin import *

###################################
## Estimate of physical parameters
###################################
rho_0 = 7850. # steel mass density 7850 kg/m^3
Deltay_0 = 0.0254 # cross sectional length, m 
Deltax_0 = 0.6 # longitudinal, symmetric axis, m 
g_0 = 9.8 # acceleration m/s^2; take 3 gs for the body force

# Create mesh
mesh = BoxMesh(0.0,0.0,0.0,0.6,0.0254,0.0254,16,10,10)

# Create function space
V = VectorFunctionSpace(mesh, 'Lagrange', 2)

# Mark boundary subdomains
left = CompiledSubDomain("near(x[0],side) && on_boundary",side = 0.0)
right = CompiledSubDomain("near(x[0],side) && on_boundary",side = 0.6)

# Dirichlet boundary condition on entire boundary
clamp = Constant((0.0, 0.0, 0.0))
bcl = DirichletBC(V, clamp, left)
bcr = DirichletBC(V, clamp, right)

bcs = [bcl, bcr]

# Create test and trial functions, and source term
u, w = TrialFunction(V), TestFunction(V)
b = Constant((0.0, rho_0*3.*g_0, 0.0))

# Elasticity parameters
E, nu = 200.0*10**9, 0.29 # 200 GPa, Gigapascal, for steel #**9
mu, lmbda = E/(2.0*(1.0+nu)), E*nu/((1.0+nu)*(1.0-2.0*nu))

# Stress
sigma = 2*mu*sym(grad(u)) + lmbda*tr(grad(u))*Identity(w.cell().geometric_dimension())

# Governing balance equation
F = inner(sigma,grad(w))*dx - dot(b,w)*dx

# Extract bilinear and linear forms from F
a, L = lhs(F), rhs(F)


# Set up PDE and solve
u = Function(V)
problem = LinearVariationalProblem(a, L, u, bcs=bcs)
solver = LinearVariationalSolver(problem)
solver.parameters["symmetric"] = True
solver.solve()


sigma_f = 2*mu*sym(grad(u)) + lmbda*tr(grad(u))*Identity(3)
T_f = sigma_f*u
T_f_u = inner( T_f, u)

# Save solution to VTK format
#File("linelast_stress.pvd","compressed") << T_f
#File("linelast_stressed.pvd","compressed") << T_f_u
# EY : 20150807 I get this error when I try to save
# TypeError: in method 'File___lshift__', argument 2 of type 'dolfin::Function const &'

#####
## normalizing u:
# I want to get the norm of the vector field u over the mesh
#####
# cf. http://fenicsproject.org/qa/4049/how-to-compute-the-norm-vector-function-each-node-vertex-fast

V_i = FunctionSpace(mesh,'Lagrange',2)
u_x = Function(V_i)
u_y = Function(V_i)
u_z = Function(V_i)
assigner_V_to_V_i = FunctionAssigner([V_i,V_i,V_i],V)
assigner_V_to_V_i.assign([u_x,u_y,u_z],u)
u_x.vector().axpy(1,u_y.vector())
u_x.vector().axpy(1,u_z.vector())

# When running these following 2 commands, I get this error:
# linelast01.py:123: RuntimeWarning: invalid value encountered in sqrt
# But when I run them in the interactive prompt, no error was received
"""
u_x.vector().set_local( np.sqrt(u_x.vector().get_local()) )
u_x.vector().apply('')
"""


# Now I want the norm squared |u|^2
u_xsq = Function(V_i)
u_ysq = Function(V_i)
u_zsq = Function(V_i)
assigner_V_to_V_isq = FunctionAssigner([V_i,V_i,V_i],V)
assigner_V_to_V_isq.assign([u_xsq,u_ysq,u_zsq],u)
u_xsq.vector().axpy(1,u_ysq.vector())
u_xsq.vector().axpy(1,u_zsq.vector())
u_xsq.vector().apply('')

plot(T_f_u/u_xsq, interactive=True)


"""
Possibly useful plotting commands from before:

# Save solution to VTK format
# File("linelast.pvd", "compressed") << u


# Plot and hold solution 
# plot(u, mode = "displacement", interactive=True)


# Save solution to VTK format
File("elasticity.pvd", "compressed") << u

# Save colored mesh partitions in VTK format if running in parallel
if MPI.size(mesh.mpi_comm()) > 1:
    File("partitions.pvd") << CellFunction("size_t", mesh, \
                                           MPI.rank(mesh.mpi_comm()))

# Project and write stress field to post-processing file
W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
stress = project(sigma(u), V=W)
File("stress.pvd") << stress

# Plot solution
plot(u, interactive=True)
"""

"""
Commands that weren't useful in the end
T_f_normed = T_f/norm(u,mesh=mesh)
T_f_u_normed = T_f_u/norm(u,mesh=mesh)**2

"""
