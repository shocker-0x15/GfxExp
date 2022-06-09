#include "../../common_shader.h"

GLSL_UNIFORM_BINDING(0, 0, sampler2D, u_srcTexture);
GLSL_UNIFORM(1, float, u_brightnessScale);
GLSL_UNIFORM(2, uint, u_flags);

GLSL_FRAGMENT_OUTPUT(0, vec4, outputF4);

float sRGB_calcLuminance(vec3 color) {
    return 0.2126729f * color.x + 0.7151522f * color.y + 0.0721750f * color.z;
}

void main(void) {
    vec3 color = texelFetch(u_srcTexture, ivec2(gl_FragCoord.xy), 0).xyz;
    bool useToneMap = ((u_flags >> 0) & 0x1) != 0;
    if (useToneMap) {
        float lum = sRGB_calcLuminance(color);
        if (lum > 0.0f) {
            float lumT = 1 - exp(-u_brightnessScale * lum);
            // simple tone-map
            color = color * (lumT / lum);
        }
        else {
            color = vec3(0.0f);
        }
    }
    outputF4 = vec4(color, 1.0f);
}
