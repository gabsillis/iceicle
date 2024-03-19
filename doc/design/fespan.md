# fespan
`fespan` is a non-owning view of finite element data inspired by `std::mdspan`. 

## Definitions 
* **Degree of Freedom (dof)**: represents an index corresponding to a basis function. 
A global degree of freedom (gdof) represents the index of that basis function in the set of all basis functions in the finite element space. A local degree of freedom (ldof) represents the index of that basis function in the set of basis functions pertaining to single element. When not specified, it is assumed that a dof refers to the local degree of freedom
* **Vector Component**: At each degree of freedom, fespan can store a vector of data,
the component represents the index into this vector.

* **Index Space**: An index space represents the set of possible indices or multi indices with a valid map to a value in the `fespan`
* **Global Index Space**: The global index space $\mathcal{I}_g$ is a set of indices $i\in\mathbb{Z}$ such that the indices $i$ form a bijection with values stored in the memory being viewed.
The values that these indices map to are required to be contiguous (this requirement doesn't put a strict requirement on $i\in\mathcal{I}_g$, rather the data it maps to). 
This is for efficient set-all-values and copy operations.
* **Local Index Space**: The local index space $\mathcal{I}_{l}$ is the set of index triples (`ielem`, `idof`, `iv`) which have a valid map to the global index space $\mathcal{I}_g$.
Note that the valid indices for `idof` can vary from element to element.
* **Global/Local Degree of Freedom Index Space**: 
The same as above except the global bijection is with global basis function indices (gdof) and the local space omits the vector component `iv`.
## Structures
### Dof Map
There are two structures to map ldofs to gdofs. The primary function of these maps is to map a pair (`ielem`, `idof`)
of the element index and local degree of freedom index respectively to a global degree of freedom index.

Dof Maps are heavy types, so the views keep const references to Dof Maps through the extents. For multiple memory spaces, this will require extra care.
#### Methods
`ndof_el` - get the number of local degrees of freedom given an element index


#### `dg_dof_map`
Maps local DG degrees of freedom to a global index through an internal `std::vector` of index offsets
## Usage

