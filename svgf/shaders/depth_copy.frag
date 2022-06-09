#include "../../common_shader.h"

GLSL_UNIFORM_BINDING(0, 0, sampler2D, u_depthTexture);

GLSL_FRAGMENT_OUTPUT(0, float, depth);

void main(void) {
    ivec2 pixIdx = ivec2(gl_FragCoord.xy);
    depth = texelFetch(u_depthTexture, pixIdx, 0).x;
}
