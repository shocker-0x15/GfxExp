#include "draw_g_buffers_shared.h"

float pow2(float x) {
    return x * x;
}

void main(void) {
    // http://john-chapman-graphics.blogspot.com/2013/01/per-object-motion-blur.html
    vec2 motionVectorInNDC = v_positionInClip.xy / v_positionInClip.z -
                             v_prevPositionInClip.xy / v_prevPositionInClip.z;
    vec2 motionVectorInScreen = 0.5f * motionVectorInNDC;
    motionVectorInScreen.y *= -1; // for upper left origin.

    vec2 prevScreenPos = gl_FragCoord.xy / u_screenSize - motionVectorInScreen;

    gBuffer_positionInWorld_texCoord_x = vec4(v_positionInWorld, v_texCoord.x);
    vec3 normalInWorld = v_normalInWorld;
    // if (dot(u_cameraPosition - v_positionInWorld, v_normalInWorld) < 0)
    //     normalInWorld = -normalInWorld; // double-sided
    normalInWorld = normalize(normalInWorld);
    if (any(isnan(normalInWorld)))
        normalInWorld = vec3(0.0f, 0.0f, 0.0f);

    vec3 tangentInWorld = v_tangentInWorld;
    tangentInWorld -= dot(tangentInWorld, normalInWorld) * normalInWorld;
    tangentInWorld = normalize(tangentInWorld);

    vec3 bitangentInWorld = cross(normalInWorld, tangentInWorld);

    vec3 normalInTangent;
    uint bumpMapMode = (u_flags >> 0) & 0x3;
    if (bumpMapMode == 0) {
        normalInTangent = 2 * texture(u_normalTexture, v_texCoord).xyz - 1;
    }
    else if (bumpMapMode == 1) {
        vec2 texValue = texture(u_normalTexture, v_texCoord).xy;
        texValue = 2.0f * texValue - 1.0f;
        float z = sqrt(1.0f - pow2(texValue.x) - pow2(texValue.y));
        normalInTangent = vec3(texValue.x, texValue.y, z);
    }
    else {
        normalInTangent = vec3(0, 0, 1);
    }
    mat3 matW2T = mat3(tangentInWorld, bitangentInWorld, normalInWorld);
    normalInWorld = normalize(matW2T * normalInTangent);

    gBuffer_normalInWorld_texCoord_y = vec4(normalInWorld, v_texCoord.y);
    gBuffer_prevScreenPos_instSlot_matSlot = vec4(
        prevScreenPos,
        uintBitsToFloat(u_instSlot),
        uintBitsToFloat(u_materialSlot));
}
