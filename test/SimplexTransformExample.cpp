#include <iceicle/transformations/SimplexElementTransformation.hpp>
#include <fstream>
#include <matplot/matplot.h>
#include <iostream>
#include <cmath>

using namespace ELEMENT::TRANSFORMATIONS;
using namespace std;
using namespace matplot;
using Point2D = MATH::GEOMETRY::Point<double, 2>;

Point2D vortex_transform(Point2D original){
    double dt = 0.01;
    Point2D pt_moved = original;
    for(int k = 0; k < 100; ++k){
        double x = pt_moved[0];
        double y = pt_moved[1];
        pt_moved[0] +=  cos(x)*sin(y) * dt;
        pt_moved[1] += -sin(x)*cos(y) * dt;
    }
    return pt_moved;
}

/** @brief plot and label the points in 2d reference domain */
void example1(){
    SimplexElementTransformation<double, int, 2, 3> trans{};

    const Point2D *reference_nodes = trans.reference_nodes();

    std::vector<double> x{};
    std::vector<double> y{};
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        x.push_back(reference_nodes[inode][0]);
        y.push_back(reference_nodes[inode][1]);
    }

    string poins_str = trans.print_ijk_poin();
    std::cout << poins_str << endl;

    /*
    scatter(x, y);
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        text(x[inode], y[inode], to_string(inode));
    }
    show();
    */
}

/**
 * @brief move a reference domain with the taylor-green vortex
 * and compare with a dense point cloud
 */
void example2(){
    SimplexElementTransformation<double, int, 2, 3> trans{};

    const Point2D *reference_nodes = trans.reference_nodes();

    std::vector<double> x{};
    std::vector<double> y{};
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        x.push_back(reference_nodes[inode][0]);
        y.push_back(reference_nodes[inode][1]);
    }

    string poins_str = trans.print_ijk_poin();
    std::cout << poins_str << endl;

    std::cout << endl << "Press any key to continue..." << endl;
    cin.get();

    // make a curved triangley boi with vortex
    std::vector<Point2D> points{};
    std::vector<int> node_numbers{};
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        Point2D pt = trans.reference_nodes()[inode];
        pt = vortex_transform(pt);
        points.push_back(pt);
        node_numbers.push_back(inode);
    }

    SimplexElementTransformation<double, int, 2, 20> dense_trans;
    x.clear();
    y.clear();
    for(int inode = 0; inode < dense_trans.nnodes(); ++inode){
        Point2D xi = dense_trans.reference_nodes()[inode];
        Point2D xpt;
        trans.transform(points, node_numbers.data(), xi, xpt);
        x.push_back(xpt[0]);
        y.push_back(xpt[1]);
    }

    vector<double> x_act{};
    vector<double> y_act{};
    for(int inode = 0; inode < dense_trans.nnodes(); ++inode){
        Point2D xpt = vortex_transform(dense_trans.reference_nodes()[inode]);
        x_act.push_back(xpt[0]);
        y_act.push_back(xpt[1]);
    }

    scatter(x, y);
    hold(on);
    scatter(x_act, y_act);
    show();
}

int main(int argc, char *argv[]){
    example2();
}
