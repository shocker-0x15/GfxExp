#version 400
#extension GL_ARB_explicit_uniform_location : enable // required in version lower 4.3
#extension GL_ARB_shading_language_420pack : enable // required in version lower 4.2

layout(location = 0) uniform uvec2 imageSize;
layout(location = 1, binding = 0) uniform sampler2D srcTexture;
layout(location = 2) uniform int flags;
layout(location = 3) uniform float brightness;

out vec4 colorF4;

float sRGB_calcLuminance(vec3 color) {
    return 0.2126729f * color.x + 0.7151522f * color.y + 0.0721750f * color.z;
}

void main(void) {
    vec2 srcPixel = gl_FragCoord.xy;
    srcPixel.y = imageSize.y - srcPixel.y;
    vec3 color = texelFetch(srcTexture, ivec2(srcPixel), 0).xyz;
    bool useToneMap = ((flags >> 0) & 0x1) != 0;
    if (useToneMap) {
        float lum = sRGB_calcLuminance(color);
        if (lum > 0.0f) {
            float lumT = 1 - exp(-brightness * lum);
            // simple tone-map
            color = color * (lumT / lum);
        }
        else {
            color = vec3(0.0f);
        }
    }
    colorF4 = vec4(color, 1.0f);
}
