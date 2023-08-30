#include <iceicle/transformations/SimplexElementTransformation.hpp>
#include <fstream>
#include <matplot/matplot.h>
int main(int argc, char *argv[]){
    using namespace ELEMENT::TRANSFORMATIONS;
    using namespace std;
    using namespace matplot;
    using Point = MATH::GEOMETRY::Point<double, 2>;
    SimplexElementTransformation<double, int, 2, 3> trans{};

    const Point *reference_nodes = trans.reference_nodes();

    std::vector<double> x{};
    std::vector<double> y{};
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        x.push_back(reference_nodes[inode][0]);
        y.push_back(reference_nodes[inode][1]);
    }

    scatter(x, y);
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        text(x[inode], y[inode], to_string(inode));
    }
    show();
    return 0;
    

    // make a curved triangley boi
    std::vector<Point> points = {
        {1.0, 0.0},
        {0.7, 0.45},
        {0.66, -0.038},
        {0.417,0.75},
        {0.297, 0.292},
        {0.33, -0.059},
        {-0.038, 1.023},
        {0.055, 0.691},
        {-0.05, 0.35},
        {0.0, 0.0}
    };

    std::vector<Point> points_ref = {
        {1 ,0},
        {0.666667, 0.333333},
        {0.666667, 0},
        {0.333333, 0.666667},
        {0.333333, 0.333333},
        {0.333333, 0},
        {0 ,1 },
        {0 ,0.666667},
        {0 ,0.333333},
        {0 ,0}
    };
    std::vector<int> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    ofstream out2("SimplexTransformTest_DiscretizedVis.dat");
    SimplexElementTransformation<double, int, 2, 20> dense_trans;
    for(int i = 0; i < dense_trans.nnodes(); ++i){
        Point ref = dense_trans.reference_nodes()[i]; // use the denser simplex to visualize the points curving
        Point phys;
        trans.transform(points, indices, ref, phys);
        out2 << phys[0] << " " << phys[1] << std::endl;
    }
}
