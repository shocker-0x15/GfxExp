#include "../../common_shader.h"

void main(void) {
    // 0: -3.0, -1.0
    // 1:  1.0, -1.0
    // 2:  1.0,  3.0
    gl_Position = vec4(
        -3.0 + 4 * ((gl_VertexID + 1) / 2), 
        -1.0 + 4 * (gl_VertexID / 2), 
        0.0, 1.0);
}
