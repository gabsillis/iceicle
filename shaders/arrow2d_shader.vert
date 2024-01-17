R"(
#version 330 core
layout (location = 0) in vec3 anchor;
layout (location = 1) in vec3 arrow;

out vertex_data {
    vec3 anchor;
    vec3 arrow;
} v_out;

void main(){
    v_out.anchor = vec3(-0.5, -0.5, 0.0);
    v_out.arrow = vec3(1.0, 1.0, 0.0);
}

)"
