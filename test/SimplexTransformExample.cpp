#include <iceicle/transformations/SimplexElementTransformation.hpp>
#include <fstream>
#include <matplot/matplot.h>
#include <iostream>
#include <cmath>

using namespace ELEMENT::TRANSFORMATIONS;
using namespace std;
using namespace matplot;
using Point2D = MATH::GEOMETRY::Point<double, 2>;
using Point3D = MATH::GEOMETRY::Point<double, 3>;

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

Point3D vortex_transform(Point3D original){
    double dt = 0.01;
    Point3D pt_moved = original;
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

    scatter(x, y);
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        text(x[inode], y[inode], to_string(inode));
    }
    show();
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

/** @brief show the transformation on a curved trace */
void example3(){
    SimplexElementTransformation<double, int, 2, 2> trans{};
    SimplexTraceTransformation<double, int, 2, 2> trace_trans{};
    std::vector<Point2D> nodes = {
        {0.0, 0.0}, // 0
        {1.0, 0.0}, // 1
        {2.0, 0.0}, // 2
        {0.0, 1.0}, // 3
        {1.4, 1.4}, // 4
        {2.0, 1.0}, // 5
        {0.0, 2.0}, // 6
        {1.0, 2.0}, // 7
        {2.0, 2.0}  // 8
    };

    std::vector<int> idxs1 = 
        { 2, 6, 0, 4, 3, 1 };
    std::vector<int> idxs2 = 
        { 2, 8, 6, 5, 7, 4 };

    int traceNrL = 2;
    int traceNrR = 1;

    std::vector<double> xL{};
    std::vector<double> yL{};
    std::vector<double> xR{};
    std::vector<double> yR{};
    double ds = 0.1;
    for(int ipoin = 0; ipoin < 11; ++ipoin){
        MATH::GEOMETRY::Point<double, 1> s = {ipoin * ds};
        Point2D xiL, xiR;
        trace_trans.transform(idxs1.data(), idxs2.data(), traceNrL, traceNrR, s, xiL, xiR);
        Point2D xptL, xptR;
        trans.transform(nodes, idxs1.data(), xiL, xptL);
        trans.transform(nodes, idxs2.data(), xiR, xptR);
        xL.push_back(xptL[0]);
        yL.push_back(xptL[1]);
        xR.push_back(xptR[0]);
        yR.push_back(xptR[1]);
    }

    scatter(xL, yL);
    hold(on);
    scatter(xR, yR);


    for(int inode = 0; inode < 11; ++inode){
        text(xL[inode], yL[inode] + 0.1, to_string(inode));
        text(xR[inode], yR[inode] - 0.1, to_string(inode));
    }
    show();
}

/**
 * @brief move a reference domain with the taylor-green vortex
 * and compare with a dense point cloud
 */
void example4(){
    SimplexElementTransformation<double, int, 3, 3> trans{};

    const Point3D *reference_nodes = trans.reference_nodes();

    std::vector<double> x{};
    std::vector<double> y{};
    std::vector<double> z{};
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        x.push_back(reference_nodes[inode][0]);
        y.push_back(reference_nodes[inode][1]);
        z.push_back(reference_nodes[inode][2]);
    }

    string poins_str = trans.print_ijk_poin();
    std::cout << poins_str << endl;

    std::cout << endl << "Press any key to continue..." << endl;
    cin.get();

    // make a curved triangley boi with vortex
    std::vector<Point3D> points{};
    std::vector<int> node_numbers{};
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        Point3D pt = trans.reference_nodes()[inode];
        pt = vortex_transform(pt);
        points.push_back(pt);
        node_numbers.push_back(inode);
    }

    SimplexElementTransformation<double, int, 3, 10> dense_trans;
    x.clear();
    y.clear();
    z.clear();
    for(int inode = 0; inode < dense_trans.nnodes(); ++inode){
        Point3D xi = dense_trans.reference_nodes()[inode];
        Point3D xpt;
        trans.transform(points, node_numbers.data(), xi, xpt);
        x.push_back(xpt[0]);
        y.push_back(xpt[1]);
        z.push_back(xpt[2]);
    }

    vector<double> x_act{};
    vector<double> y_act{};
    vector<double> z_act{};
    for(int inode = 0; inode < dense_trans.nnodes(); ++inode){
        Point3D xpt = vortex_transform(dense_trans.reference_nodes()[inode]);
        x_act.push_back(xpt[0]);
        y_act.push_back(xpt[1]);
        z_act.push_back(xpt[2]);
    }

    scatter3(x, y, z);
    hold(on);
    scatter3(x_act, y_act, z_act);
    show();
}


void example5(){
    SimplexElementTransformation<double, int, 3, 2> trans{};
    SimplexTraceTransformation<double, int, 3, 2> trace_trans{};
    std::vector<Point3D> nodes = {
        {0.0, 0.0, 0.0}, // 0
        {1.0, 0.0, 0.0}, // 1
        {1.0, 1.0, 0.0}, // 2
        {0.0, 1.0, 0.0}, // 3
        {0.0, 0.0, 1.0}, // 4
        {0.5, 0.0, 0.0}, // 5
        {1.0, 0.5, 0.0}, // 6
        {0.5, 1.0, 0.0}, // 7
        {0.0, 0.5, 0.0}, // 8
        {0.5, 0.5, 0.0}, // 9
        {0.0, 0.0, 0.5}, // 10
        {0.7, 0.0, 0.5}, // 11
        {0.8, 0.8, 0.5}, // 12
        {0.5, 0.5, 0.5}, // 13
    };

    std::vector<int> idxs1 = 
        { 1, 3, 4, 0, 9, 13, 11, 5, 8, 10 };

    std::vector<int> idxs2 = 
        { 1, 2, 3, 4, 6, 7, 9, 11, 12, 13 };

    int traceNrL = 3;
    int traceNrR = 1;

    std::vector<double> xL{};
    std::vector<double> yL{};
    std::vector<double> zL{};
    std::vector<double> xR{};
    std::vector<double> yR{};
    std::vector<double> zR{};

    SimplexElementTransformation<double, int, 2, 8> trace_domain{};

    for(int ipoin = 0; ipoin < trace_domain.nnodes(); ++ipoin){
        MATH::GEOMETRY::Point<double, 2> s = trace_domain.reference_nodes()[ipoin];
        Point3D xiL, xiR;
        trace_trans.transform(idxs1.data(), idxs2.data(), traceNrL, traceNrR, s, xiL, xiR);
        Point3D xptL, xptR;
        trans.transform(nodes, idxs1.data(), xiL, xptL);
        trans.transform(nodes, idxs2.data(), xiR, xptR);
        xL.push_back(xptL[0]);
        yL.push_back(xptL[1]);
        zL.push_back(xptL[2]);
        xR.push_back(xptR[0]);
        yR.push_back(xptR[1]);
        zR.push_back(xptR[2]);
    }

    scatter3(xL, yL, zL);
    hold(on);
    //scatter3(xR, yR, zL);

/*
    for(int inode = 0; inode < trace_domain.nnodes(); ++inode){
        text(xL[inode], yL[inode] + 0.1, zL[inode], to_string(inode));
        text(xR[inode], yR[inode] - 0.1, zL[inode], to_string(inode));
    }
*/
    show();
}

int main(int argc, char *argv[]){
    example4();
}
