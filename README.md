# MegaTE

Code repository for ACM SIGCOMM '24 paper 'MegaTE: Extending WAN Traffic Engineering to Millions of Endpoints in Virtualized Cloud'.

Run `download.sh` to fetch the traffic matrices and pre-computed paths used in
our evaluation. (For confidentiality reasons, we only share TMs and paths for
topologies from the Internet Topology Zoo.)


## Dependencies

- Setup validated on Ubuntu 16.04.
- Python 3.6 (Anaconda installation recommended)
  - See `environment.yml` for a list of Python library dependencies
- Gurobi 8.1.1 (Requires a Gurobi license)


## Code structure 
```
.
|-- benchmarks              # test code for MegaTE, NcFlow and LP-all
|   |-- demo                # demo for small-scale example
|-- lib                     # source code for MegaTE SSP, NcFlow and LP
|-- scripts                 # generating site/server level topologies, paths and traffic matrices 
|-- topologies              # network topologies(site-level, server-level)
|   |-- paths               # paths in topologies(site-level, server-level)
|   |-- site-level topologies    
|   |-- server-level topologies         
|-- traffic-matrices        # traffic demand matrices(site-level/server-level, split/non-split)
    |-- site-level traffic matrices    
    |-- server-level traffic matrices
```
