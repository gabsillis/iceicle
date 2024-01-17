R"(
#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vertex_data {
    vec3 anchor;
    vec3 arrow;
} gs_in[];

// The ratio of the shaft width to the length vec
const float SHAFT_WIDTH_MUL = 0.1;

// The ratio of the widest point of the head to vec
const float HEAD_WIDTH_MUL = 0.2;

// The ratio of the length of the shaft to vec
const float SHAFT_LENGTH_MUL = 0.8;

void create_shaft(vec3 position, vec3 arrow){
    vec3 normal = vec3(-arrow[1], arrow[0], 0.0);
    
    // bottom right
    gl_Position = vec4(position - normal * 0.5 * SHAFT_WIDTH_MUL, 0.0);
    EmitVertex();

    // top right 
    gl_Position = vec4(position - normal * 0.5 * SHAFT_WIDTH_MUL + arrow * (SHAFT_LENGTH_MUL), 0.0);
    EmitVertex();

    // bottom left
    gl_Position = vec4(position + normal * 0.5 * SHAFT_WIDTH_MUL, 0.0);
    EmitVertex();

    // top left 
    gl_Position = vec4(position + normal * 0.5 * SHAFT_WIDTH_MUL + arrow * (SHAFT_LENGTH_MUL), 0.0);
    EmitVertex();

    EndPrimitive();
}

void create_head(vec3 position, vec3 arrow){

    vec3 normal = vec3(-arrow[1], arrow[0], 0.0);

    // top point
    gl_Position = vec4(position + arrow, 0.0);
    EmitVertex();

    // bottom right
    gl_Position = vec4(position + arrow * SHAFT_LENGTH_MUL - normal * 0.5 * HEAD_WIDTH_MUL, 0.0);
    EmitVertex();

    // bottom left
    gl_Position = vec4(position + arrow * SHAFT_LENGTH_MUL + normal * 0.5 * HEAD_WIDTH_MUL, 0.0);
    EmitVertex();

    EndPrimitive();
}

void main() {
    create_shaft(gs_in[0].anchor, gs_in[0].arrow);
    //create_head(gs_in[0].anchor, gs_in[0].arrow);

}

)"
