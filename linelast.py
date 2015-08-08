## linelast.py
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

from dolfin import *

# Create mesh
mesh = UnitCubeMesh(8,8,8)

# Create function space
V = VectorFunctionSpace(mesh, 'Lagrange', 2)

# Create test and trial functions, and source term
u, w = TrialFunction(V), TestFunction(V)
b = Constant((1.0, 0.0, 0.0))

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = E/(2.0*(1.0+nu)), E*nu/((1.0+nu)*(1.0-2.0*nu))

# Stress
sigma = 2*mu*sym(grad(u)) + lmbda*tr(grad(u))*Identity(w.cell().geometric_dimension())

# Governing balance equation
F = inner(sigma,grad(w))*dx - dot(b,w)*dx

# Extract bilinear and linear forms from F
a, L = lhs(F), rhs(F)

# Dirichlet boundary condition on entire boundary
c = Constant((0.0, 1.0, 0.0))
bc = DirichletBC(V, c, DomainBoundary())

# Set up PDE and solve
u = Function(V)
problem = LinearVariationalProblem(a, L, u, bcs=bc)
solver = LinearVariationalSolver(problem)
solver.parameters["symmetric"] = True
solver.solve()


# Save solution to VTK format
File("linelast.pvd", "compressed") << u


# Plot and hold solution 
plot(u, mode = "displacement", interactive=True)


"""
# Plot and hold solution
plot(u, mode = "displacement", interactive = True)
"""


"""
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
