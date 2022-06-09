#include "draw_g_buffers_shared.h"

void main(void) {
    vec4 prevPositionInWorld = u_prevMatM2W * vec4(a_position, 1.0f);
    vec4 positionInWorld = u_matM2W * vec4(a_position, 1.0f);

    v_prevPositionInClip = (u_prevMatW2C * prevPositionInWorld).xyw;
    v_positionInClip = (u_matW2C * positionInWorld).xyw;

    v_positionInWorld = positionInWorld.xyz;
    v_tangentInWorld = mat3(u_matM2W) * a_tangent;
    v_normalInWorld = u_nMatM2W * a_normal;
    v_texCoord = a_texCoord;

    gl_Position = u_matW2CWithOffset * positionInWorld;
}
