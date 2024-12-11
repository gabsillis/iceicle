/// @brief Mechanism for restarting from previous state
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include "iceicle/anomaly_log.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include <iomanip>
#include <iceicle/iceicle_mpi_utils.hpp>
#include <filesystem>
#include <fstream>

namespace iceicle {
   
    /// @brief write the restart file 
    /// NOTE: assumes restart will happen with equivalent toplogy
    /// and discretization
    ///
    /// @brief fespace the finite element space 
    /// @brief u the solution to write to restart file 
    /// @brief k the iteration identifier for the restart file
    template<class T, class IDX, int ndim, class LayoutPolicy>
    auto write_restart(
        FESpace<T, IDX, ndim>& fespace,
        fespan<T, LayoutPolicy> u,
        IDX k
    ) -> void {

        std::filesystem::path restart_directory{std::filesystem::current_path()};
        restart_directory /= "RESTART";

        // create the path if it doesn't exist
        std::filesystem::create_directories(restart_directory);

        // get MPI rank
        IDX my_pid = iceicle::mpi::mpi_world_rank();

        // open file 
        std::filesystem::path out_filename = (restart_directory / ("restart" + std::to_string(k) 
                    + "_" + std::to_string(my_pid)));
        std::ofstream out{out_filename};
        out << std::setprecision(16);

        if(!out) { 
            util::AnomalyLog::log_anomaly(util::Anomaly{
                "Cannot open restart file for writing", util::general_anomaly_tag{}}); 
        }

        // write out the nodes
        auto& nodes{fespace.meshptr->coord};
        for(auto& node : nodes){
            for(int idim = 0; idim < ndim; ++idim) 
                { out << node[idim] << " "; }
        }
        out << std::endl;

        // write out the solution vector
        for(IDX ielem = 0; ielem < fespace.elements.size(); ++ielem){
            for(IDX idof = 0; idof < u.ndof(ielem); ++idof){
                for(int iv = 0; iv < u.nv(); ++iv){
                    out << u[ielem, idof, iv] << " ";
                }
            }
        }
    }

    /// @brief read a restart file
    ///
    /// @param fespace the finite element space (will overwrite mesh nodes)
    /// @param u the solution to read 
    /// @param restart_name the name of the restart files omitting the "_processid"
    /// the names will have _<id> where <id> is the id of each process 
    /// just provide the name that preceeds this
    template<class T, class IDX, int ndim, class LayoutPolicy>
    auto read_restart(
        FESpace<T, IDX, ndim>& fespace,
        fespan<T, LayoutPolicy> u,
        std::string restart_name
    ) -> void {

        std::filesystem::path restart_directory{std::filesystem::current_path()};
        restart_directory /= "RESTART";


        // get MPI rank
        IDX my_pid = iceicle::mpi::mpi_world_rank();

        // open file 
        std::filesystem::path in_filename = (restart_directory / (restart_name 
                    + "_" + std::to_string(my_pid)));
        std::ifstream in{in_filename};

        // read in the nodes
        auto& nodes{fespace.meshptr->coord};
        for(auto& node : nodes){
            for(int idim = 0; idim < ndim; ++idim) 
                { in >> node[idim]; }
        }
        fespace.meshptr->update_coord_els();

        // write out the solution vector
        for(IDX ielem = 0; ielem < fespace.elements.size(); ++ielem){
            for(IDX idof = 0; idof < u.ndof(ielem); ++idof){
                for(int iv = 0; iv < u.nv(); ++iv){
                    in >> u[ielem, idof, iv];
                }
            }
        }
    }

}
