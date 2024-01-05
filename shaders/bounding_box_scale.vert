R"(
#version 330 core

layout(location = 0) in vec3 model_pos;

uniform vec3 xmin;
uniform vec3 xmax;

void main() {
    // get the sizes in each direction and then the max
    // of the bounding box
    vec3 size = xmax - xmin;
    float size_max = max(max(size[0], size[1]), size[2]);

    // get a normalized coordinate t in the bounding box
    vec3 t = model_pos - xmin;
    t.x = 2 * t.x / size_max - 1;
    t.y = 2 * t.y / size_max - 1;
    t.z = 2 * t.z / size_max - 1;

    // adjusted position
    vec3 normpos;
    normpos.x = (1 - size.x / size_max) + t.x;
    normpos.y = (1 - size.y / size_max) + t.y;
    normpos.z = (1 - size.z / size_max) + t.z;

    const float BORDER = 0.1;
    const float VIEWPORT_SIZE = 2.0 - 2 * BORDER;
    const float VIEWPORT_HALF = 1.0 - BORDER;

    vec3 t2;
    t2.x = (normpos.x + 1.0) / 2;
    t2.y = (normpos.y + 1.0) / 2;
    t2.z = (normpos.z + 1.0) / 2;

    gl_Position.x = -VIEWPORT_HALF  + t2.x * VIEWPORT_SIZE;
    gl_Position.y = -VIEWPORT_HALF  + t2.y * VIEWPORT_SIZE;
    gl_Position.z = -VIEWPORT_HALF  + t2.z * VIEWPORT_SIZE;
    gl_Position.w = 1.0;
}
)"

